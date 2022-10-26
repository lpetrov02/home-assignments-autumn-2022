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


def solve_pnp(intrinsic_matrix, frame_number, calculated_points, corner_storage):
    frame_corners = corner_storage[frame_number]
    image_points = dict(zip(frame_corners.ids, frame_corners.points))
    actual_points = []
    actual_ids = []
    actual_image_points = []
    for corner_id in frame_corners.ids:
        if corner_id in calculated_points:
            actual_points.append(calculated_points[corner_id])
            actual_ids.append(corner_id)
            actual_image_points.append(image_points[corner_id])
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(actual_points, actual_image_points, intrinsic_matrix)
    view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
    inliers_ids = np.array(actual_ids)[inliers]

    return view_mat, inliers_ids


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # TODO: implement
    # INITIALIZATION WITH TWO KNOWN VIEWS
    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    points3d, correspondences_ids, median_cos = triangulate_correspondences(correspondences,
                                                                            pose_to_view_mat3x4(known_view_1[1]),
                                                                            pose_to_view_mat3x4(known_view_2[1]),
                                                                            intrinsic_mat,
                                                                            TriangulationParameters(1000, 0, 0))
    frame_count = len(corner_storage)
    view_mats = [np.zeros(known_view_1[1].shape)] * frame_count
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    point_cloud_builder = PointCloudBuilder(correspondences_ids, points3d)

    processed_frames = [False] * (frame_count + 1)
    processed_frames[known_view_1[0]] = True
    processed_frames[known_view_2[0]] = True
    processed_frames[frame_count] = True

    processed_frames_set = {known_view_1[0], known_view_2[0]}

    # NOW WE HAVE TO PROCESS ALL FRAMES OF THE VIDEO
    while len(processed_frames_set) != frame_count:
        previous_processed = -1
        for fn, is_processed in enumerate(processed_frames):
            if is_processed and fn < frame_count:
                new_fn = fn // 2 if previous_processed == -1 else (fn + previous_processed) // 2
                parent_fn = fn
            elif fn == frame_count:
                new_fn = (fn + previous_processed) // 2
                parent_fn = previous_processed
            else:
                continue

            if not processed_frames[new_fn]:
                processed_frames_set.add(new_fn)
            new_view_mat, new_inliers = solve_pnp(intrinsic_mat, new_fn, point_cloud_builder.points, corner_storage)
            view_mats[new_fn] = new_view_mat

            parent_frame_view = view_mats[parent_fn]
            new_correspondences = build_correspondences(corner_storage[parent_fn], corner_storage[new_fn])
            new_points3d, new_correspondences_ids, new_median_cos = triangulate_correspondences(new_correspondences,
                                                                                                parent_frame_view,
                                                                                                new_view_mat,
                                                                                                intrinsic_mat,
                                                                                                TriangulationParameters(
                                                                                                    1000, 0, 0))
            point_cloud_builder.add_points(new_correspondences_ids, new_points3d)
            processed_frames[new_fn] = True
            previous_processed = fn

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
