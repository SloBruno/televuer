"""Each TeleVuer session loop removes vuer's default-grid once, before upserts.

Fakes only (no vuer server, no network): vuer/cv2 are stubbed, the session
records events, and the infinite loop is stopped from asyncio.sleep.
"""
import asyncio
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


class _Stop(Exception):
    pass


class _At:
    def __init__(self, fn):
        self.fn = fn

    def __matmul__(self, arg):
        return self.fn(arg)


class FakeSession:
    def __init__(self):
        self.events = []

    @property
    def remove(self):
        return _At(lambda keys: self.events.append(("Remove", tuple(keys))))

    def upsert(self, element, to=None, **kwargs):
        self.events.append(("Upsert", to))


def _load(monkeypatch):
    vuer = types.ModuleType("vuer")
    vuer.Vuer = object
    schemas = types.ModuleType("vuer.schemas")
    for name in ("ImageBackground", "Hands", "MotionControllers", "WebRTCVideoPlane", "WebRTCStereoVideoPlane"):
        setattr(schemas, name, lambda *a, **k: object())
    monkeypatch.setitem(sys.modules, "vuer", vuer)
    monkeypatch.setitem(sys.modules, "vuer.schemas", schemas)
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    monkeypatch.syspath_prepend(str(SRC))
    sys.modules.pop("televuer.televuer", None)
    try:
        return importlib.import_module("televuer.televuer")
    finally:
        sys.modules.pop("televuer.televuer", None)


LOOPS = ["main_image_binocular_zmq", "main_image_monocular_zmq", "main_image_binocular_webrtc",
         "main_image_monocular_webrtc", "main_image_binocular_zmq_ego", "main_image_monocular_zmq_ego",
         "main_image_binocular_webrtc_ego", "main_image_monocular_webrtc_ego", "main_pass_through"]


@pytest.mark.parametrize("loop_name", LOOPS)
@pytest.mark.parametrize("hand_tracking", [True, False])
def test_loop_removes_default_grid_once_then_keeps_upserting(monkeypatch, loop_name, hand_tracking):
    module = _load(monkeypatch)
    tv = module.TeleVuer.__new__(module.TeleVuer)
    tv.use_hand_tracking = hand_tracking
    tv.img_shape = (4, 8, 3)
    tv.img_height, tv.img_width = 4, 4
    tv.img2display = np.zeros((4, 8, 3), np.uint8)
    tv.display_fps = 30.0
    tv.webrtc_url = "https://x/offer"
    tv.distance_to_camera = 1.0
    tv.image_height = 1.0
    tv.aspect_ratio = 1.0
    tv.binocular = "binocular" in loop_name
    session = FakeSession()
    sleeps = {"n": 0}

    async def fake_sleep(_s):
        sleeps["n"] += 1
        if sleeps["n"] >= 3:
            raise _Stop

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    with pytest.raises(_Stop):
        asyncio.run(getattr(tv, loop_name)(session))

    removes = [e for e in session.events if e[0] == "Remove"]
    assert removes == [("Remove", ("default-grid",))]
    assert session.events[0] == ("Remove", ("default-grid",))     # before the first upsert
    assert session.events[1] == ("Upsert", "bgChildren")          # hands/controllers still added
    if loop_name != "main_pass_through":  # pass-through loop only sleeps
        assert len([e for e in session.events if e[0] == "Upsert"]) >= 2


def test_remove_failure_does_not_break_loop(monkeypatch):
    module = _load(monkeypatch)
    tv = module.TeleVuer.__new__(module.TeleVuer)

    class Broken:
        @property
        def remove(self):
            raise RuntimeError("no remove")

    tv._remove_default_grid(Broken())  # must not raise
