"""No middleware, no robot sockets: exercise the actual ZMQ scene producer."""
import asyncio
from collections import deque
from types import SimpleNamespace
import time

import numpy as np

from televuer.televuer import TeleVuer
import televuer.televuer as core


class Transport:
    def __init__(self):
        self.buffer = 0

    def get_write_buffer_size(self):
        return self.buffer


class Session:
    def __init__(self):
        self.transport = Transport()
        self.socket = SimpleNamespace(closed=False, _writer=SimpleNamespace(transport=self.transport))
        self.uplink_queue = deque(maxlen=3)
        self.events = []

    def upsert(self, elements, **kwargs):
        self.events.append((elements, kwargs))


def viewer():
    tv = TeleVuer.__new__(TeleVuer)
    tv.use_hand_tracking = False
    tv.img2display = np.zeros((480, 1280, 3), dtype=np.uint8)
    tv.img_width = 640
    tv.aspect_ratio = 4 / 3
    tv.display_fps = 100
    tv.video_plane_height = 1.0
    tv.video_plane_distance = 1.0
    return tv


def test_slow_stereo_encoding_does_not_stall_control_event_loop(monkeypatch):
    def slow_image(*args, **kwargs):
        time.sleep(0.06)
        return kwargs
    monkeypatch.setattr(core, "ImageBackground", slow_image)

    async def run():
        tv, session = viewer(), Session()
        task = asyncio.create_task(tv.main_image_binocular_zmq(session))
        gaps = []
        start = previous = time.monotonic()
        try:
            while time.monotonic() - start < 0.35:
                await asyncio.sleep(0.005)
                now = time.monotonic()
                gaps.append(now - previous)
                previous = now
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert len(session.events) >= 2  # controllers plus an image pair
        assert max(gaps) < 0.045, gaps
    asyncio.run(run())


def test_congested_session_skips_encoding_and_keeps_scene_messages():
    from televuer.video_sender import VideoFrameSender
    async def run():
        session = Session()
        sender = VideoFrameSender(session)
        def must_not_encode():
            raise AssertionError('congested video must not encode')
        session.uplink_queue.append(b'controller-scene')
        assert not await sender.publish(must_not_encode)
        assert list(session.uplink_queue) == [b'controller-scene']
        session.uplink_queue.clear()
        session.transport.buffer = 65536
        assert not await sender.publish(must_not_encode)
        assert not session.events
        assert sender.stats['skipped'] == 2
        session.transport.buffer = 0
        assert await sender.publish(lambda: ['latest-image'])
        assert session.events == [(['latest-image'], {'to': 'bgChildren'})]
    asyncio.run(run())


def test_disconnect_during_encoding_discards_pair_without_session_error():
    from televuer.video_sender import VideoFrameSender
    import threading
    async def run():
        session = Session()
        sender = VideoFrameSender(session)
        started, finish = threading.Event(), threading.Event()
        def encode():
            started.set()
            assert finish.wait(1)
            return ['obsolete']
        task = asyncio.create_task(sender.publish(encode))
        while not started.is_set():
            await asyncio.sleep(0.001)
        session.socket.closed = True
        finish.set()
        assert not await task
        assert not sender.connected()
        assert not session.events
    asyncio.run(run())


def test_removed_vuer_session_is_not_a_crash():
    from televuer.video_sender import VideoFrameSender
    class Removed:
        @property
        def socket(self):
            raise KeyError('session removed')
    async def run():
        sender = VideoFrameSender(Removed())
        assert not sender.connected()
        assert not await sender.publish(lambda: [])
    asyncio.run(run())
