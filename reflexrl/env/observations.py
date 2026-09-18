"""Pixel preprocessing shared by the student, the teacher dataset and eval.

The student sees a stack of downsampled RGB frames. The teacher sees the
last full-resolution frames. Both derive from the same rendered screen, so
teacher labels attach to exactly the observation the student would see.
"""

from __future__ import annotations

import cv2
import numpy as np

FULL_H, FULL_W = 180, 320  # render resolution (RES_320X180)
STUDENT_H, STUDENT_W = 64, 112
STUDENT_STACK = 4


def to_student_frame(frame: np.ndarray) -> np.ndarray:
    """(FULL_H, FULL_W, 3) uint8 -> (3, STUDENT_H, STUDENT_W) uint8, channels-first."""
    small = cv2.resize(frame, (STUDENT_W, STUDENT_H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small.transpose(2, 0, 1))


def stack_frames(frames: list[np.ndarray]) -> np.ndarray:
    """K channels-first frames -> (K*3, H, W) uint8 observation."""
    return np.concatenate(frames, axis=0)
