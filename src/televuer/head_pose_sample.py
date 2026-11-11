"""Atomic validated headset-pose sample (matrix + monotonic receipt time).

Vuer sends the headset pose as a 16-element column-major matrix, either as
``CAMERA_MOVE`` (``{camera: {matrix}}`` on older clients, ``{matrix}`` on the
locally served 0.0.60 client) or as hosted-client ``HEAD_MOVE``
(``{matrix}``). The matrix and its receipt timestamp are published together
so a consumer can never pair a fresh timestamp with an older pose. Invalid
input never advances the timestamp.
"""

import numpy as np


HEAD_POSE_SAMPLE_SIZE = 17  # 4x4 column-major + monotonic timestamp
_SE3_ATOL = 1e-4            # browser matrices are float32


def extract_event_matrix(value):
    """Return the raw matrix from either supported payload shape, else None."""
    if not isinstance(value, dict):
        return None
    matrix = value.get("matrix")
    if matrix is None:
        camera = value.get("camera")
        if isinstance(camera, dict):
            matrix = camera.get("matrix")
    return matrix


def _as_rigid_pose(matrix):
    try:
        flat = np.asarray(matrix, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if flat.shape != (16,) or not np.isfinite(flat).all():
        return None
    pose = flat.reshape(4, 4, order="F")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=_SE3_ATOL):
        return None
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_SE3_ATOL):
        return None
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_SE3_ATOL):
        return None
    return flat


def write_head_pose_sample(shared, matrix, timestamp):
    """Publish a valid rigid pose with its timestamp; reject anything else."""
    flat = _as_rigid_pose(matrix)
    if flat is None:
        return False
    try:
        timestamp = float(timestamp)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(timestamp) or timestamp <= 0.0:
        return False
    with shared.get_lock():
        shared[:16] = flat
        shared[16] = timestamp
    return True


def read_head_pose_sample(shared):
    """Return ``(4x4 pose, monotonic timestamp)`` from one locked read."""
    with shared.get_lock():
        values = np.array(shared[:], dtype=float)
    return values[:16].reshape(4, 4, order="F"), float(values[16])
