import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "televuer" / "arm_pose_source.py"
SPEC = importlib.util.spec_from_file_location("arm_pose_source", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load arm_pose_source test module")
ARM_POSE_SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARM_POSE_SOURCE)
select_arm_pose_samples = ARM_POSE_SOURCE.select_arm_pose_samples


class SelectArmPoseSamplesTest(unittest.TestCase):
    def test_controller_source_uses_controller_poses_even_when_hand_tracking_is_available(self):
        left_hand = np.full((4, 4), 1.0)
        right_hand = np.full((4, 4), 2.0)
        left_controller = np.full((4, 4), 3.0)
        right_controller = np.full((4, 4), 4.0)

        selected_left, selected_right, source_uses_openxr_hand_convention = select_arm_pose_samples(
            arm_pose_source="controller",
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            left_controller_pose=left_controller,
            right_controller_pose=right_controller,
        )

        np.testing.assert_array_equal(selected_left, left_controller)
        np.testing.assert_array_equal(selected_right, right_controller)
        self.assertFalse(source_uses_openxr_hand_convention)

    def test_hand_source_preserves_existing_hand_pose_convention(self):
        left_hand = np.full((4, 4), 1.0)
        right_hand = np.full((4, 4), 2.0)
        left_controller = np.full((4, 4), 3.0)
        right_controller = np.full((4, 4), 4.0)

        selected_left, selected_right, source_uses_openxr_hand_convention = select_arm_pose_samples(
            arm_pose_source="hand",
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            left_controller_pose=left_controller,
            right_controller_pose=right_controller,
        )

        np.testing.assert_array_equal(selected_left, left_hand)
        np.testing.assert_array_equal(selected_right, right_hand)
        self.assertTrue(source_uses_openxr_hand_convention)

    def test_unknown_source_is_rejected_before_pose_processing(self):
        pose = np.eye(4)
        with self.assertRaisesRegex(ValueError, "arm_pose_source"):
            select_arm_pose_samples(
                arm_pose_source="unknown",
                left_hand_pose=pose,
                right_hand_pose=pose,
                left_controller_pose=pose,
                right_controller_pose=pose,
            )


if __name__ == "__main__":
    unittest.main()
