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


def _viewer_with_head_state(module):
    viewer = module.TeleVuer.__new__(module.TeleVuer)
    viewer.head_pose_shared = Array("d", 16, lock=True)
    viewer.head_pose_timestamp_shared = Value("d", 0.0, lock=True)
    viewer.head_pose_source_shared = Array("c", 16, lock=True)
    viewer.client_info_shared = Array("c", 512, lock=True)
    return viewer


def test_head_move_real_schema_updates_column_major_pose_and_timestamp(monkeypatch):
    module = _import_televuer(monkeypatch)
    viewer = _viewer_with_head_state(module)
    matrix = list(range(16))
    event = types.SimpleNamespace(value={"matrix": matrix})
    monkeypatch.setattr(module.time, "monotonic", lambda: 123.5)

    asyncio.run(viewer.on_head_move(event, None))

    np.testing.assert_array_equal(viewer.head_pose, np.array(matrix).reshape(4, 4, order="F"))
    assert viewer.head_pose_timestamp == 123.5
    assert viewer.head_pose_source == "HEAD_MOVE"


def test_camera_move_accepts_legacy_nested_and_direct_matrix_schemas(monkeypatch):
    module = _import_televuer(monkeypatch)
    viewer = _viewer_with_head_state(module)
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.0)
    first = list(range(16))
    second = list(range(16, 32))

    asyncio.run(viewer.on_cam_move(types.SimpleNamespace(value={"camera": {"matrix": first}}), None))
    np.testing.assert_array_equal(viewer.head_pose, np.array(first).reshape(4, 4, order="F"))
    assert viewer.head_pose_source == "CAMERA_MOVE"

    asyncio.run(viewer.on_cam_move(types.SimpleNamespace(value={"matrix": second}), None))
    np.testing.assert_array_equal(viewer.head_pose, np.array(second).reshape(4, 4, order="F"))


def test_init_records_actual_browser_version_and_head_capability(monkeypatch, capsys):
    module = _import_televuer(monkeypatch)
    viewer = _viewer_with_head_state(module)
    monkeypatch.setattr(module.TeleVuer, "_vuer_package_version", staticmethod(lambda: "0.0.60"))
    event = types.SimpleNamespace(value={
        "client": "browser", "pkg": "@vuer-ai/vuer", "pkgVersion": "0.0.103",
        "userAgent": "QuestBrowser/42",
    })

    asyncio.run(viewer.on_client_init(event, None))

    assert "pkg=@vuer-ai/vuer pkgVersion=0.0.103; HEAD_MOVE=enabled" in capsys.readouterr().out
    assert viewer.client_info == {
        "client": "browser",
        "pkg": "@vuer-ai/vuer",
        "pkgVersion": "0.0.103",
        "userAgent": "QuestBrowser/42",
        "serverPackageVersion": "0.0.60",
        "headComponentExpected": True,
    }


def test_init_without_version_reports_head_disabled(monkeypatch):
    module = _import_televuer(monkeypatch)
    viewer = _viewer_with_head_state(module)

    asyncio.run(viewer.on_client_init(types.SimpleNamespace(value={}), None))

    assert viewer.client_info["pkgVersion"] is None
    assert viewer.client_info["headComponentExpected"] is False


def test_init_upserts_head_only_for_known_modern_hosted_client(monkeypatch):
    module = _import_televuer(monkeypatch)

    class Session:
        def __init__(self):
            self.components = []

        def upsert(self, component, **kwargs):
            self.components.append(component)

    viewer = _viewer_with_head_state(module)
    session = Session()

    # Scenes render before metadata negotiation; only a known supported client
    # receives the Head tag later through this session.
    asyncio.run(viewer.on_client_init(types.SimpleNamespace(value={
        "client": "browser", "pkg": "@vuer-ai/vuer", "pkgVersion": "0.0.103",
    }), session))

    assert [item.serialize() for item in session.components] == [{
        "tag": "Head", "key": "head_tracking", "stream": True, "fps": 30,
        "show": False,
    }]


def test_init_never_upserts_head_for_local_old_or_unknown_client(monkeypatch):
    module = _import_televuer(monkeypatch)

    class Session:
        def __init__(self):
            self.components = []

        def upsert(self, component, **kwargs):
            self.components.append(component)

    viewer = _viewer_with_head_state(module)
    for value in (
        {"client": "browser", "pkg": "@vuer-ai/vuer", "pkgVersion": "0.0.60"},
        {"client": "browser", "pkg": "@vuer-ai/vuer", "pkgVersion": "broken"},
        {"client": "browser", "pkg": "vuer", "pkgVersion": "9.9.9"},
        {},
    ):
        session = Session()
        asyncio.run(viewer.on_client_init(types.SimpleNamespace(value=value), session))
        assert session.components == [], value


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
