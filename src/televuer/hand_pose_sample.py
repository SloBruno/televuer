"""Atomic validated hand-skeleton wrist-pose handoff for arm IK."""

import numpy as np


HAND_POSE_SAMPLE_SIZE = 33  # left 4x4, right 4x4, monotonic timestamp
_SE3_ATOL = 1e-5


def is_valid_wrist_pose(pose) -> bool:
    """Return whether ``pose`` is a finite rigid 4×4 homogeneous transform."""
    try:
        matrix = np.asarray(pose, dtype=float).reshape(4, 4, order="F")
    except (TypeError, ValueError):
        return False
    if not np.isfinite(matrix).all():
        return False
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=_SE3_ATOL):
        return False
    rotation = matrix[:3, :3]
    return (
        np.allclose(rotation.T @ rotation, np.eye(3), atol=_SE3_ATOL)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=_SE3_ATOL)
    )


def write_hand_pose_sample(shared, left_pose, right_pose, timestamp):
    """Atomically write a valid pair; invalid input leaves the prior sample intact."""
    if not (is_valid_wrist_pose(left_pose) and is_valid_wrist_pose(right_pose)):
        return False
    if not np.isfinite(timestamp) or float(timestamp) <= 0.0:
        return False
    with shared.get_lock():
        shared[:16] = np.asarray(left_pose, dtype=float).reshape(16, order="F")
        shared[16:32] = np.asarray(right_pose, dtype=float).reshape(16, order="F")
        shared[32] = float(timestamp)
    return True


def read_hand_pose_sample(shared):
    """Return a self-consistent left/right hand wrist pair and timestamp."""
    with shared.get_lock():
        values = np.array(shared[:], dtype=float)
    return (
        values[:16].reshape(4, 4, order="F"),
        values[16:32].reshape(4, 4, order="F"),
        float(values[32]),
    )
