import importlib.util
import multiprocessing
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "televuer" / "controller_pose_sample.py"
SPEC = importlib.util.spec_from_file_location("controller_pose_sample", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load controller_pose_sample test module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ControllerPoseSampleTest(unittest.TestCase):
    def test_pose_pair_and_timestamp_round_trip_as_one_sample(self):
        shared = multiprocessing.Array("d", MODULE.CONTROLLER_POSE_SAMPLE_SIZE, lock=True)
        left = np.eye(4)
        left[:3, 3] = (-0.2, 1.1, -0.3)
        right = np.eye(4)
        right[:3, 3] = (0.2, 1.1, -0.3)

        self.assertTrue(MODULE.write_controller_pose_sample(shared, left, right, 42.5))
        observed_left, observed_right, observed_timestamp = MODULE.read_controller_pose_sample(shared)

        np.testing.assert_array_equal(observed_left, left)
        np.testing.assert_array_equal(observed_right, right)
        self.assertEqual(observed_timestamp, 42.5)

    def test_invalid_pose_pair_does_not_advance_freshness(self):
        shared = multiprocessing.Array("d", MODULE.CONTROLLER_POSE_SAMPLE_SIZE, lock=True)
        left = np.eye(4)
        right = np.eye(4)
        self.assertTrue(MODULE.write_controller_pose_sample(shared, left, right, 10.0))

        invalid_samples = []
        nan_pose = np.eye(4)
        nan_pose[0, 0] = np.nan
        invalid_samples.append((nan_pose, right, 11.0))
        singular_pose = np.eye(4)
        singular_pose[2, 2] = 0.0
        invalid_samples.append((left, singular_pose, 12.0))
        malformed_pose = np.eye(4)
        malformed_pose[3] = (0.0, 0.0, 1.0, 1.0)
        invalid_samples.append((malformed_pose, right, 13.0))
        invalid_samples.append((left, right, float("nan")))

        for invalid_left, invalid_right, timestamp in invalid_samples:
            self.assertFalse(
                MODULE.write_controller_pose_sample(
                    shared, invalid_left, invalid_right, timestamp
                )
            )
            observed_left, observed_right, observed_timestamp = MODULE.read_controller_pose_sample(shared)
            np.testing.assert_array_equal(observed_left, left)
            np.testing.assert_array_equal(observed_right, right)
            self.assertEqual(observed_timestamp, 10.0)


if __name__ == "__main__":
    unittest.main()
