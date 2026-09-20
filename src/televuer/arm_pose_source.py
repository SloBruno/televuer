"""Select the XR pose stream that owns G1 arm IK targets."""


def select_arm_pose_samples(
    *,
    arm_pose_source,
    left_hand_pose,
    right_hand_pose,
    left_controller_pose,
    right_controller_pose,
):
    """Return left/right raw XR poses and their initial-pose convention.

    Controller poses already use Unitree's arm initial-pose convention. Hand
    skeleton wrist poses use OpenXR's hand convention and need the existing
    per-side conversion in ``TeleVuerWrapper``.
    """
    if arm_pose_source == "hand":
        return left_hand_pose, right_hand_pose, True
    if arm_pose_source == "controller":
        return left_controller_pose, right_controller_pose, False
    raise ValueError(f"Unknown arm_pose_source: {arm_pose_source!r}")
