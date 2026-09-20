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
        left = np.arange(16, dtype=float).reshape(4, 4, order="F")
        right = np.arange(16, 32, dtype=float).reshape(4, 4, order="F")

        MODULE.write_controller_pose_sample(shared, left, right, 42.5)
        observed_left, observed_right, observed_timestamp = MODULE.read_controller_pose_sample(shared)

        np.testing.assert_array_equal(observed_left, left)
        np.testing.assert_array_equal(observed_right, right)
        self.assertEqual(observed_timestamp, 42.5)


if __name__ == "__main__":
    unittest.main()
