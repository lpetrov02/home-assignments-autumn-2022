#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
import matplotlib.pyplot as plt


from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class Corner:
    def __init__(self, i, j, c_id=0):
        self._i = i
        self._j = j
        self._id = c_id

    def __hash__(self):
        return hash(f"{self._i},{self._j}")


def draw_corners(img, corners, radius=3, color=(0, 1, 0),
                 y_first=True):
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        rgb = img.copy()
    else:
        raise ValueError('Illegal image format')
    for pt in corners:
        if y_first:
            pt_tuple = tuple(pt[::-1])
        else:
            pt_tuple = tuple(pt)
        cv2.circle(rgb, pt_tuple, radius, color)
    return rgb


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    opt_flow_params = dict(
        winSize=(25, 25), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    image_0 = np.uint8(frame_sequence[0] * 255)
    corners_coordinates = cv2.goodFeaturesToTrack(image_0, 200, 0.01, 10).reshape(-1, 2)
    corners_ids = np.arange(corners_coordinates.shape[0])
    next_id = corners_coordinates.shape[0]
    corners = FrameCorners(
        corners_ids,
        corners_coordinates,
        np.ones(corners_coordinates.shape[0]) * 15
    )
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_1 = np.uint8(image_1 * 255)
        new_corners, new_ids = [], []
        old_corners, status, err = cv2.calcOpticalFlowPyrLK(
            image_0, image_1, corners_coordinates, None, **opt_flow_params
        )
        old_corners = old_corners.reshape((-1, 2)).astype(int)
        mask = np.array([False] * image_1.shape[0] * image_1.shape[1]).reshape(image_1.shape)
        for i, cor in enumerate(old_corners):
            if status[i, 0]:
                new_ids.append(corners_ids[i])
                new_corners.append(cor)
                mask[cor[1], cor[0]] = True
        new_ids = np.array(new_ids)
        new_corners = np.array(new_corners)

        new_corners_2 = cv2.goodFeaturesToTrack(image_1, 200, 0.01, 10, mask=mask, blockSize=(10, 10)).reshape(-1, 2)
        new_ids_2 = np.arange(next_id, next_id + new_corners_2.shape[0])
        next_id += new_corners_2.shape[0]

        corners_coordinates = np.vstack([new_corners, new_corners_2])
        corners_ids = np.hstack([new_ids, new_ids_2])
        corners = FrameCorners(
            corners_ids,
            corners_coordinates,
            np.ones(corners_coordinates.shape[0]) * 10
        )
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
