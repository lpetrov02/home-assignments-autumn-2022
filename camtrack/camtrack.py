#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np
from queue import PriorityQueue

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    TriangulationParameters,
    build_correspondences,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    rodrigues_and_translation_to_view_mat3x4,
    triangulate_correspondences,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose
)


def solve_pnp(intrinsic_matrix, frame_number, calculated_points, calculated_points_ids, corner_storage):
    frame_corners = corner_storage[frame_number]
    image_points = dict(zip(frame_corners.ids.reshape(-1), frame_corners.points))
    calc_points = dict(zip(calculated_points_ids.reshape(-1), calculated_points))
    common_ids = np.array(list(set(calculated_points_ids.reshape(-1)) & set(frame_corners.ids.reshape(-1))))
    actual_points = []
    actual_image_points = []
    for corner_id in common_ids:
        actual_points.append(calc_points[corner_id])
        actual_image_points.append(image_points[corner_id])
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(actual_points), np.array(actual_image_points),
                                                     intrinsic_matrix, np.zeros(4))
    view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
    inliers_ids = np.array(common_ids)[inliers]

    return view_mat, inliers_ids


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()
    if known_view_1[0] > known_view_2[0]:
        known_view_1, known_view_2 = known_view_2, known_view_1

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats = [None] * frame_count
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    points3d, correspondences_ids, median_cos = triangulate_correspondences(correspondences,
                                                                            pose_to_view_mat3x4(known_view_1[1]),
                                                                            pose_to_view_mat3x4(known_view_2[1]),
                                                                            intrinsic_mat,
                                                                            TriangulationParameters(1000, 0, 0))

    point_cloud_builder = PointCloudBuilder(correspondences_ids, points3d)
    print(f"First two frames processed: {known_view_1[0]} and {known_view_2[0]}")
    print(f"3D points on the 1st step: {len(points3d)}, point cloud size: {len(points3d)}")

    processed_frames_set = {known_view_1[0], known_view_2[0]}

    # NOW WE HAVE TO PROCESS ALL FRAMES OF THE VIDEO
    shift = 20
    added_points_on_previous_epoch = 2
    min_common_points = 4
    direction_parameter = {"frw": {"dir": 1, "end": frame_count}, "back": {"dir": -1, "end": -1}}
    while len(processed_frames_set) != frame_count:
        for fn in list(processed_frames_set):
            for d in direction_parameter:
                new_fn = fn + shift * direction_parameter[d]["dir"]
                while new_fn * direction_parameter[d]["dir"] < \
                        direction_parameter[d]["end"] * direction_parameter[d]["dir"]:
                    if new_fn in processed_frames_set:
                        new_fn += 1 * direction_parameter[d]["dir"]
                        continue

                    new_view_mat, new_inliers = solve_pnp(intrinsic_mat, new_fn, point_cloud_builder.points,
                                                          point_cloud_builder.ids, corner_storage)
                    # print(new_view_mat, new_inliers, "\n\n\n")
                    parent_frame_view_mat = view_mats[fn]
                    new_correspondences = build_correspondences(corner_storage[fn], corner_storage[new_fn])
                    new_points3d, new_correspondences_ids, new_median_cos = \
                        triangulate_correspondences(new_correspondences,
                                                    parent_frame_view_mat,
                                                    new_view_mat,
                                                    intrinsic_mat,
                                                    TriangulationParameters(
                                                        1000, 0, 0))
                    if len(new_correspondences_ids) < min_common_points:
                        break

                    print(f"Processed frame: {new_fn}. Inliers: {len(new_inliers)}. "
                          f"Point cloud size: {len(point_cloud_builder.points)}")
                    print(f"Triangulated on this step: {len(new_points3d)}.")
                    print(f"Frames processed: {len(processed_frames_set)} out of {frame_count} \n\n")

                    view_mats[new_fn] = new_view_mat
                    point_cloud_builder.add_points(new_correspondences_ids, new_points3d)
                    processed_frames_set.add(new_fn)
                    new_fn += 1 * direction_parameter[d]["dir"]
        if len(processed_frames_set) == added_points_on_previous_epoch:
            shift //= 2

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
