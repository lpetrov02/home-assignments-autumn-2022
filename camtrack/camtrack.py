#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp
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
    to_opencv_camera_mat4x4,
    view_mat3x4_to_pose
)


def solve_pnp(intrinsic_matrix, frame_number, calculated_points, calculated_points_ids, corner_storage, excluded):
    frame_corners = corner_storage[frame_number]
    image_points = dict(zip(frame_corners.ids.reshape(-1), frame_corners.points))
    calc_points = dict(zip(calculated_points_ids.reshape(-1), calculated_points))
    common_ids_set = set(calculated_points_ids.reshape(-1)) & set(frame_corners.ids.reshape(-1)) - excluded
    if len(common_ids_set) < 4:
        return None, None

    common_ids = list(common_ids_set)
    actual_points = []
    actual_image_points = []
    for corner_id in common_ids:
        actual_points.append(calc_points[corner_id])
        actual_image_points.append(image_points[corner_id])
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(actual_points), np.array(actual_image_points),
                                                     intrinsic_matrix, np.zeros(4))
    view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
    inliers_ids = np.array(common_ids)[inliers]
    excluded |= (common_ids_set - set(inliers_ids.reshape(-1).tolist()))

    return view_mat, inliers_ids


def rmat_and_tvec_to_mat4x4(rmat, tvec):
    mat = np.eye(4)
    mat[:3, :3] = rmat
    mat[:3, 3] = tvec
    return mat


def build_extended_correspondences(corners_list):
    ids = None
    for corners in corners_list:
        ids = ids & set(corners.ids.reshape(-1).tolist()) if ids is not None else set(corners.ids.reshape(-1).tolist())
    return np.array(list(ids)), \
           [
               corners_list[i].points[np.in1d(corners_list[i].ids.reshape(-1), list(ids))]
               for i in range(len(corners_list))
           ]


def triangulate_n_points(points2d, camera_poses, proj_mat):
    """
    points2d:
    [[(x11, y11), (x12, y12)],  - точки для первого кадра
     [(x21, y21), (x22, y22)]]  - точки для второго кадра
    """
    points = np.array(points2d)
    P = [proj_mat @ np.linalg.inv(rmat_and_tvec_to_mat4x4(camera_pose[:, :3], camera_pose[:, 3]))
         for camera_pose in camera_poses]
    res_points = []
    for i in range(len(points[0])):
        A = np.vstack([[P[j][3, :] * points[j, i, 0] - P[j][0, :], P[j][3, :] * points[j, i, 1] - P[j][1, :]]
                       for j in range(len(points))])
        X = np.linalg.lstsq(A[:, :3] / A[:, 3].reshape(-1, 1), -np.ones(A.shape[0]), rcond=0)
        res_points.append(X[0])
    return np.array(res_points)


def initialize_prim(corners_1, corners_2, K):
    ids_1 = corners_1.ids.flatten()
    ids_2 = corners_2.ids.flatten()
    _, (indices_1, indices_2) = snp.intersect(ids_1, ids_2, indices=True)
    points_1 = corners_1.points[indices_1].reshape(-1, 2)
    points_2 = corners_2.points[indices_2].reshape(-1, 2)

    if points_1.shape[0] < 5:
        return False, None, None, None, None, None

    E_base, mask_base = cv2.findEssentialMat(points_1, points_2, K, method=cv2.RANSAC)
    H, mask_homo = cv2.findHomography(points_1, points_2, method=cv2.RANSAC)

    if mask_base.mean() > mask_homo.mean():
        retval, R, t, mask, _ = cv2.recoverPose(
            E_base, points1=points_1, points2=points_2, cameraMatrix=K, distanceThresh=50
        )
        return True, R, t, mask_base, mask_base.mean() - mask_homo.mean(), mask_base.sum()
    return False, None, None, None, None, None


def initialize(intrinsic_mat, corner_storage, min_inliers_count=5, good_inliers_count=950):
    best_res = {"R": np.eye(3), "t": np.zeros(3), "n": (0, 1)}
    best_inliers_count = 0

    fc = len(corner_storage)
    f1 = fc // 2 - min(fc // 10, 10)
    f2 = fc // 2 + min(fc // 10, 10)

    while f2 - f1 > 0:
        res, R, t, mask, inliers_part, inliers_count =\
            initialize_prim(corner_storage[f1], corner_storage[f2], intrinsic_mat)
        print(f1, f2, inliers_count)
        if res and inliers_count >= min_inliers_count:
            if inliers_count > best_inliers_count:
                best_res["R"], best_res["t"] = R, t.flatten()
                best_res["n"] = (f1, f2)
                best_inliers_count = inliers_count
            if inliers_count > good_inliers_count:
                print(f1, f2)
            return best_res
        if abs(fc // 2 - f1) > abs(fc // 2 - f2):
            f1 += 1
        else:
            f2 -= 1
        continue
    return best_res


'''
def initialize(intrinsic_mat, corner_storage, min_inliers_part=(1.0 - 1e-6), min_inliers_count=5):
    best_inliers_part = 0.0
    best_res = {"R": np.eye(3), "t": np.zeros(3), "n": (0, 1)}
    best_inliers_count = 0

    fc = len(corner_storage)
    d = max(fc // 5, 20)
    while d > 10:
        f1, f2 = 0, d - 1
        while f2 < fc:
            res, R, t, mask, inliers_part, inliers_count =\
                initialize_prim(corner_storage[f1], corner_storage[f2], intrinsic_mat)
            if not res:
                f1, f2 = f2, f2 + d
                continue
            if inliers_count >= max(best_inliers_count, min_inliers_count):
                best_res["R"], best_res["t"] = R, t.flatten()
                best_res["n"] = (f1, f2)
                best_inliers_count = inliers_count
            f1, f2 = f1 + max(1, fc // 100), f2 + max(1, fc // 100)
        if best_inliers_count > 10:
            return best_res
        d = d * 4 // 5
    return best_res
'''


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
    proj_mat = to_opencv_camera_mat4x4(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats = [None] * frame_count
    # view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    # view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    # correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    # points3d, correspondences_ids, median_cos =\
    #     triangulate_correspondences(
    #         correspondences,
    #         pose_to_view_mat3x4(known_view_1[1]),
    #         pose_to_view_mat3x4(known_view_2[1]),
    #         intrinsic_mat,
    #         TriangulationParameters(50, 0, 0)
    #     )
    #
    # point_cloud_builder = PointCloudBuilder(correspondences_ids, points3d)
    # print(f"First two frames processed: {known_view_1[0]} and {known_view_2[0]}")
    # print(f"3D points on the 1st step: {len(points3d)}, point cloud size: {len(points3d)}")

    view = initialize(intrinsic_mat, corner_storage)
    view_mats[view["n"][0]] = pose_to_view_mat3x4(Pose(np.eye(3), -np.zeros(3)))
    view_mats[view["n"][1]] = pose_to_view_mat3x4(Pose(view["R"], -view["R"] @ view["t"]))
    correspondences = build_correspondences(corner_storage[view["n"][0]], corner_storage[view["n"][1]])
    points3d, correspondences_ids, median_cos =\
        triangulate_correspondences(
            correspondences,
            pose_to_view_mat3x4(Pose(np.eye(3), -np.zeros(3))),
            pose_to_view_mat3x4(Pose(view["R"], -view["R"] @ view["t"])),
            intrinsic_mat,
            TriangulationParameters(50, 0, 0)
        )

    point_cloud_builder = PointCloudBuilder(correspondences_ids, points3d)
    print(f"First two frames processed: {view['n'][0]} and {view['n'][1]}")
    print(f"3D points on the 1st step: {len(points3d)}, point cloud size: {len(points3d)}")

    processed_frames_set = {view["n"][0], view["n"][1]}

    # NOW WE HAVE TO PROCESS ALL FRAMES OF THE VIDEO
    shift = frame_count
    min_angle = 10
    added_points_on_previous_epoch = 2
    min_common_points = 2
    direction_parameter = {"frw": {"dir": 1, "end": frame_count}, "back": {"dir": -1, "end": -1}}
    outliers = set()
    steps = 0
    was_processed_on_step = [-1 for _ in range(frame_count)]
    long_period = frame_count * 3 // 4
    while len(processed_frames_set) != frame_count:
        for fn in list(processed_frames_set):
            # re-triangulation block
            try:
                if fn in processed_frames_set and steps - was_processed_on_step[fn] > long_period:
                    new_correspondences_n = 0
                    left_shift, right_shift = fn, frame_count - fn - 1
                    new_correspondences_ids, new_correspondences_points = None, None
                    while new_correspondences_n < 5 and left_shift * right_shift > 0:
                        new_correspondences_ids, new_correspondences_points = build_extended_correspondences(
                            [corner_storage[fn - left_shift],
                             corner_storage[fn],
                             corner_storage[fn + right_shift]]
                        )
                        new_correspondences_n = len(new_correspondences_ids)
                        if new_correspondences_n < 5:
                            left_shift, right_shift = left_shift * 9 // 10, right_shift * 9 // 10
                    if new_correspondences_n >= 5:
                        new_points3d = triangulate_n_points(new_correspondences_points,
                                                            [
                                                                view_mats[fn],
                                                                view_mats[fn - left_shift],
                                                                view_mats[fn + right_shift]
                                                            ],
                                                            proj_mat)
                        # point_cloud_builder.add_points(new_correspondences_ids, new_points3d)
                        was_processed_on_step[fn] = steps
                        # print(f"Frame {fn} was re-triangulated")
            except:
                pass
            # end of re-triangulation block

            for d in direction_parameter:
                new_fn = fn + shift * direction_parameter[d]["dir"]
                while new_fn * direction_parameter[d]["dir"] < \
                        direction_parameter[d]["end"] * direction_parameter[d]["dir"]:
                    if new_fn in processed_frames_set:
                        new_fn += 1 * direction_parameter[d]["dir"]
                        continue

                    new_view_mat, new_inliers = solve_pnp(intrinsic_mat, new_fn, point_cloud_builder.points,
                                                          point_cloud_builder.ids, corner_storage, outliers)

                    if new_view_mat is None and new_inliers is None:
                        print(f"Fail for frame {new_fn}")
                        break

                    parent_frame_view_mat = view_mats[fn]
                    new_correspondences = build_correspondences(corner_storage[fn], corner_storage[new_fn])
                    new_points3d, new_correspondences_ids, new_median_cos = \
                        triangulate_correspondences(new_correspondences,
                                                    parent_frame_view_mat,
                                                    new_view_mat,
                                                    intrinsic_mat,
                                                    TriangulationParameters(
                                                        50, min_angle, 0))
                    if len(new_correspondences_ids) < min_common_points:
                        break

                    view_mats[new_fn] = new_view_mat
                    point_cloud_builder.add_points(new_correspondences_ids, new_points3d)
                    processed_frames_set.add(new_fn)

                    was_processed_on_step[new_fn] = steps
                    steps += 1

                    print(f"Median angle cosinus: {new_median_cos}")
                    print(f"Processed frame: {new_fn}. Inliers: {len(new_inliers)}. "
                          f"Point cloud size: {len(point_cloud_builder.points)}")
                    print(f"Triangulated on this step: {len(new_points3d)}.")
                    print(f"Frames processed: {len(processed_frames_set)} out of {frame_count}")
                    print(f"Total outliers: {len(outliers)}\n")

                    new_fn += 1 * direction_parameter[d]["dir"]
        if len(processed_frames_set) == added_points_on_previous_epoch:
            shift = shift * 3 // 4
            min_angle //= 2
        added_points_on_previous_epoch = len(processed_frames_set)

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
