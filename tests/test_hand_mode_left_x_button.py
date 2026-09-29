import types

import numpy as np

from televuer.tv_wrapper import TeleVuerWrapper


def _wrapper_with_left_x_pressed():
    wrapper = TeleVuerWrapper.__new__(TeleVuerWrapper)
    identity = np.eye(4)
    wrapper.use_hand_tracking = True
    wrapper.arm_pose_source = "controller"
    wrapper.arm_reference_mode = "head"
    wrapper.return_hand_rot_data = False
    wrapper.tvuer = types.SimpleNamespace(
        head_pose=identity,
        hand_pose_sample=(identity, identity, 1.0),
        controller_pose_sample=(identity, identity, 2.0),
        motion_data_ready=True,
        left_hand_pinch=False,
        left_hand_pinchValue=0.0,
        left_hand_squeeze=False,
        left_hand_squeezeValue=0.0,
        right_hand_pinch=False,
        right_hand_pinchValue=0.0,
        right_hand_squeeze=False,
        right_hand_squeezeValue=0.0,
        left_hand_positions=np.zeros((25, 3)),
        right_hand_positions=np.zeros((25, 3)),
        left_ctrl_trigger=False,
        left_ctrl_triggerValue=0.0,
        right_ctrl_trigger=False,
        right_ctrl_triggerValue=0.0,
        left_ctrl_thumbstick=False,
        left_ctrl_thumbstickValue=np.zeros(2),
        right_ctrl_thumbstick=False,
        right_ctrl_thumbstickValue=np.zeros(2),
        left_ctrl_aButton=True,
        left_ctrl_bButton=False,
        right_ctrl_aButton=False,
        right_ctrl_bButton=False,
    )
    return wrapper


def test_hand_tracking_mode_forwards_left_x_button_to_teledata():
    data = _wrapper_with_left_x_pressed().get_tele_data()
    assert data.left_ctrl_aButton is True
