"""Atomic controller-pose sample handoff for arm IK."""

import numpy as np


CONTROLLER_POSE_SAMPLE_SIZE = 33  # left 4x4, right 4x4, monotonic timestamp


def write_controller_pose_sample(shared, left_pose, right_pose, timestamp):
    """Publish a left/right controller pose pair and timestamp under one lock."""
    with shared.get_lock():
        shared[:16] = np.asarray(left_pose, dtype=float).reshape(16, order="F")
        shared[16:32] = np.asarray(right_pose, dtype=float).reshape(16, order="F")
        shared[32] = float(timestamp)


def read_controller_pose_sample(shared):
    """Return a self-consistent left/right controller pose pair and timestamp."""
    with shared.get_lock():
        values = np.array(shared[:], dtype=float)
    return (
        values[:16].reshape(4, 4, order="F"),
        values[16:32].reshape(4, 4, order="F"),
        float(values[32]),
    )
