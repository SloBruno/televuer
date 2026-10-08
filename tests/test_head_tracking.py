import asyncio
import importlib
import sys
import types
from multiprocessing import Array, Value

import numpy as np


def _import_televuer(monkeypatch):
    class Element:
        tag = "SceneElement"

        def __init__(self, *children, key=None, **kwargs):
            self.__dict__.update(tag=self.tag, key=key, **kwargs)
            if children:
                self.children = children

        def serialize(self):
            return dict(self.__dict__)

    class Vuer:
        def __init__(self, **kwargs):
            self.handlers = {}

        def add_handler(self, name):
            return lambda fn: self.handlers.setdefault(name, fn) or fn

        def spawn(self, start=False):
            return lambda fn: fn

    schemas = types.ModuleType("vuer.schemas")
    schemas.SceneElement = Element
    for name in ("ImageBackground", "Hands", "MotionControllers", "WebRTCVideoPlane", "WebRTCStereoVideoPlane"):
        setattr(schemas, name, type(name, (Element,), {"tag": name}))
    monkeypatch.setitem(sys.modules, "vuer", types.SimpleNamespace(Vuer=Vuer))
    monkeypatch.setitem(sys.modules, "vuer.schemas", schemas)
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(COLOR_BGR2RGB=1))
    for name in [key for key in sys.modules if key == "televuer" or key.startswith("televuer.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("televuer.televuer")


def test_python_vuer_0060_compatible_generic_head_component(monkeypatch):
    module = _import_televuer(monkeypatch)

    component = module.HeadTracking(key="head_tracking", stream=True, fps=30, show=False)

    assert component.serialize() == {
        "tag": "Head",
        "key": "head_tracking",
        "stream": True,
        "fps": 30,
        "show": False,
    }


def test_head_move_real_schema_updates_column_major_pose_and_timestamp(monkeypatch):
    module = _import_televuer(monkeypatch)
    viewer = module.TeleVuer.__new__(module.TeleVuer)
    viewer.head_pose_shared = Array("d", 16, lock=True)
    viewer.head_pose_timestamp_shared = Value("d", 0.0, lock=True)
    matrix = list(range(16))
    event = types.SimpleNamespace(value={"matrix": matrix})
    monkeypatch.setattr(module.time, "monotonic", lambda: 123.5)

    asyncio.run(viewer.on_head_move(event, None))

    np.testing.assert_array_equal(viewer.head_pose, np.array(matrix).reshape(4, 4, order="F"))
    assert viewer.head_pose_timestamp == 123.5


def test_all_scene_modes_request_hosted_client_head_stream(monkeypatch):
    module = _import_televuer(monkeypatch)

    class Session:
        def __init__(self):
            self.components = []

        def upsert(self, component, **kwargs):
            self.components.append(component)

    class Stop(Exception):
        pass

    async def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(module.asyncio, "sleep", stop)
    for scene_name in (
        "main_image_binocular_zmq", "main_image_monocular_zmq",
        "main_image_binocular_webrtc", "main_image_monocular_webrtc",
        "main_image_binocular_zmq_ego", "main_image_monocular_zmq_ego",
        "main_image_binocular_webrtc_ego", "main_image_monocular_webrtc_ego",
        "main_pass_through",
    ):
        viewer = module.TeleVuer.__new__(module.TeleVuer)
        viewer.use_hand_tracking = False
        viewer.display_fps = 30.0
        viewer.img2display = np.zeros((2, 4, 3), dtype=np.uint8)
        viewer.img_width = 2
        viewer.aspect_ratio = 1.0
        viewer.video_plane_height = 1.0
        viewer.video_plane_distance = 1.0
        viewer.webrtc_url = "https://example.invalid/offer"
        session = Session()
        try:
            asyncio.run(getattr(viewer, scene_name)(session))
        except Stop:
            pass
        heads = [item.serialize() for item in session.components if getattr(item, "tag", None) == "Head"]
        assert heads == [{"tag": "Head", "key": "head_tracking", "stream": True, "fps": 30, "show": False}], scene_name


def test_wrapper_forwards_head_receipt_timestamp_and_fallback_state(monkeypatch):
    _import_televuer(monkeypatch)
    wrapper_module = importlib.import_module("televuer.tv_wrapper")
    wrapper = wrapper_module.TeleVuerWrapper.__new__(wrapper_module.TeleVuerWrapper)
    wrapper.use_hand_tracking = False
    wrapper.arm_pose_source = "controller"
    wrapper.arm_reference_mode = "head_yaw"
    wrapper.return_hand_rot_data = False
    identity = np.eye(4)
    wrapper.tvuer = types.SimpleNamespace(
        head_pose=identity,
        head_pose_timestamp=7.5,
        hand_pose_sample=(identity, identity, 1.0),
        controller_pose_sample=(identity, identity, 2.0),
        motion_data_ready=True,
        left_ctrl_trigger=False, left_ctrl_triggerValue=0.0,
        left_ctrl_squeeze=False, left_ctrl_squeezeValue=0.0,
        left_ctrl_aButton=False, left_ctrl_bButton=False,
        left_ctrl_thumbstick=False, left_ctrl_thumbstickValue=np.zeros(2),
        right_ctrl_trigger=False, right_ctrl_triggerValue=0.0,
        right_ctrl_squeeze=False, right_ctrl_squeezeValue=0.0,
        right_ctrl_aButton=False, right_ctrl_bButton=False,
        right_ctrl_thumbstick=False, right_ctrl_thumbstickValue=np.zeros(2),
    )

    data = wrapper.get_tele_data()
    assert data.head_pose_sample_timestamp == 7.5
    assert data.head_pose_is_fallback is False

    wrapper.tvuer.head_pose = np.zeros((4, 4))
    assert wrapper.get_tele_data().head_pose_is_fallback is True
