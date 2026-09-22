import importlib.util
import multiprocessing
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "televuer" / "hand_pose_sample.py"
SPEC = importlib.util.spec_from_file_location("hand_pose_sample", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load hand_pose_sample test module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HandPoseSampleTest(unittest.TestCase):
    def test_hand_events_publish_an_atomic_wrist_pair(self):
        source = (Path(__file__).parents[1] / "src" / "televuer" / "televuer.py").read_text(encoding="utf-8")
        hand_handler = source[source.index("async def on_hand_move"):source.index("## immersive MODE")]
        self.assertIn("write_hand_pose_sample(", hand_handler)
        self.assertIn("self.hand_pose_sample_shared", hand_handler)

    def test_invalid_wrist_pair_does_not_replace_the_last_valid_sample(self):
        shared = multiprocessing.Array("d", MODULE.HAND_POSE_SAMPLE_SIZE, lock=True)
        valid_left = np.eye(4)
        valid_right = np.eye(4)
        MODULE.write_hand_pose_sample(shared, valid_left, valid_right, 10.0)

        invalid_right = np.eye(4)
        invalid_right[0, 0] = np.nan
        self.assertFalse(MODULE.write_hand_pose_sample(shared, valid_left, invalid_right, 20.0))

        observed_left, observed_right, observed_timestamp = MODULE.read_hand_pose_sample(shared)
        np.testing.assert_array_equal(observed_left, valid_left)
        np.testing.assert_array_equal(observed_right, valid_right)
        self.assertEqual(observed_timestamp, 10.0)

    def test_singular_wrist_pair_is_rejected(self):
        shared = multiprocessing.Array("d", MODULE.HAND_POSE_SAMPLE_SIZE, lock=True)
        singular_left = np.eye(4)
        singular_left[:3, :3] = 0.0

        self.assertFalse(MODULE.write_hand_pose_sample(shared, singular_left, np.eye(4), 10.0))
        self.assertEqual(MODULE.read_hand_pose_sample(shared)[2], 0.0)

    def test_pose_pair_and_timestamp_round_trip_as_one_sample(self):
        shared = multiprocessing.Array("d", MODULE.HAND_POSE_SAMPLE_SIZE, lock=True)
        left = np.eye(4)
        left[:3, 3] = (0.1, -0.2, 0.3)
        right = np.eye(4)
        right[:3, 3] = (-0.4, 0.5, -0.6)

        self.assertTrue(MODULE.write_hand_pose_sample(shared, left, right, 42.5))
        observed_left, observed_right, observed_timestamp = MODULE.read_hand_pose_sample(shared)

        np.testing.assert_array_equal(observed_left, left)
        np.testing.assert_array_equal(observed_right, right)
        self.assertEqual(observed_timestamp, 42.5)


if __name__ == "__main__":
    unittest.main()
