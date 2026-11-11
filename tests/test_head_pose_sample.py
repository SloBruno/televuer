import importlib.util
import multiprocessing
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "televuer" / "head_pose_sample.py"
SPEC = importlib.util.spec_from_file_location("head_pose_sample", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load head_pose_sample test module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _yaw_pose(yaw, xyz=(0.1, 1.6, -0.2)):
    pose = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    # OpenXR: yaw about +Y.
    pose[:3, :3] = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    pose[:3, 3] = xyz
    return pose


class HeadPoseSampleTest(unittest.TestCase):
    def _shared(self):
        return multiprocessing.Array("d", MODULE.HEAD_POSE_SAMPLE_SIZE, lock=True)

    def test_initial_sample_is_never_fresh(self):
        pose, timestamp = MODULE.read_head_pose_sample(self._shared())
        self.assertEqual(timestamp, 0.0)
        self.assertEqual(pose.shape, (4, 4))

    def test_column_major_matrix_and_timestamp_round_trip(self):
        shared = self._shared()
        pose = _yaw_pose(0.4)
        column_major = pose.reshape(16, order="F").tolist()
        self.assertTrue(MODULE.write_head_pose_sample(shared, column_major, 12.5))
        observed, timestamp = MODULE.read_head_pose_sample(shared)
        np.testing.assert_allclose(observed, pose)
        self.assertEqual(timestamp, 12.5)

    def test_float32_browser_rounding_is_accepted(self):
        shared = self._shared()
        pose = _yaw_pose(1.1).astype(np.float32).astype(float)
        self.assertTrue(MODULE.write_head_pose_sample(shared, pose.reshape(16, order="F"), 3.0))

    def test_invalid_input_never_advances_freshness(self):
        shared = self._shared()
        good = _yaw_pose(0.2)
        self.assertTrue(MODULE.write_head_pose_sample(shared, good.reshape(16, order="F"), 10.0))
        nan_pose = good.copy(); nan_pose[0, 0] = np.nan
        singular = good.copy(); singular[:3, :3] = 0.0
        scaled = good.copy(); scaled[:3, :3] *= 2.0
        bad_last_row = good.copy(); bad_last_row[3] = (0.0, 0.0, 1.0, 1.0)
        for value, ts in (
            (nan_pose.reshape(16, order="F"), 11.0),
            (singular.reshape(16, order="F"), 12.0),
            (scaled.reshape(16, order="F"), 13.0),
            (bad_last_row.reshape(16, order="F"), 14.0),
            ([1.0, 2.0, 3.0], 15.0),
            (None, 16.0),
            ("matrix", 17.0),
            (good.reshape(16, order="F"), float("nan")),
            (good.reshape(16, order="F"), 0.0),
        ):
            self.assertFalse(MODULE.write_head_pose_sample(shared, value, ts))
            observed, timestamp = MODULE.read_head_pose_sample(shared)
            np.testing.assert_allclose(observed, good)
            self.assertEqual(timestamp, 10.0)

    def test_event_matrix_extraction_accepts_both_payload_shapes(self):
        flat = list(range(16))
        self.assertEqual(MODULE.extract_event_matrix({"matrix": flat}), flat)
        self.assertEqual(MODULE.extract_event_matrix({"camera": {"matrix": flat}}), flat)
        self.assertIsNone(MODULE.extract_event_matrix({"camera": "x"}))
        self.assertIsNone(MODULE.extract_event_matrix(None))
        self.assertIsNone(MODULE.extract_event_matrix([1, 2]))


if __name__ == "__main__":
    unittest.main()
