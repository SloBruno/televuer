"""Real Vuer loopback lifecycle checks; image input is synthetic, no robot/DDS."""
import asyncio
import logging
import sys
import types
import unittest
from pathlib import Path
from importlib.metadata import version

import msgpack
import numpy as np
from aiohttp import ClientSession, web
from vuer import Vuer

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
# TeleVuer imports an external application haptics adapter; producers do not use it.
haptics = types.ModuleType('teleop.utils.haptics')
haptics.HapticTransportAdapter = object
sys.modules['teleop'] = types.ModuleType('teleop')
sys.modules['teleop.utils'] = types.ModuleType('teleop.utils')
sys.modules['teleop.utils.haptics'] = haptics
from televuer.televuer import TeleVuer
from televuer.session_lifecycle import install_socket_lifecycle


class Lifecycle(unittest.IsolatedAsyncioTestCase):
    async def start_server(self, producer, fail_send=False):
        app = Vuer(host='127.0.0.1', queue_len=3)
        install_socket_lifecycle(app)
        app.spawn(start=False)(producer)
        if fail_send:
            async def send(*args, **kwargs):
                raise RuntimeError('injected uplink failure')
            app.send = send
        app._add_route('/', app.socket_index)
        runner = web.AppRunner(app.app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        self.addAsyncCleanup(runner.cleanup)
        return app, 'http://127.0.0.1:%d/' % site._server.sockets[0].getsockname()[1]

    async def wait_for(self, condition):
        async def wait():
            while not condition():
                await asyncio.sleep(.005)
        await asyncio.wait_for(wait(), 2)

    async def test_all_producers_two_clients_disconnect_independently(self):
        self.assertEqual(version('vuer'), '0.0.60')
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        errors = []
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        self.addCleanup(loop.set_exception_handler, previous)
        modes = ['main_image_monocular_zmq', 'main_image_binocular_zmq',
                 'main_image_monocular_zmq_ego', 'main_image_binocular_zmq_ego',
                 'main_image_monocular_webrtc', 'main_image_binocular_webrtc',
                 'main_image_monocular_webrtc_ego', 'main_image_binocular_webrtc_ego',
                 'main_pass_through']
        for mode in modes:
            with self.subTest(mode=mode):
                viewer = TeleVuer.__new__(TeleVuer)
                viewer.use_hand_tracking = True
                viewer.display_fps = 50
                viewer.webrtc_url = 'http://127.0.0.1/video'
                viewer.aspect_ratio = 1.0
                viewer.img_width = 16
                viewer.img2display = np.full((16, 32 if 'binocular' in mode else 16, 3), 128, dtype=np.uint8)
                viewer.video_plane_height = 1.0
                viewer.video_plane_distance = 1.0
                workers = {}
                async def producer(session):
                    workers[session.CURRENT_WS_ID] = asyncio.current_task()
                    await getattr(viewer, mode)(session)
                app, url = await self.start_server(producer)
                async with ClientSession() as client:
                    a = await client.ws_connect(url)
                    b = await client.ws_connect(url)
                    await self.wait_for(lambda: len(workers) == 2)
                    ids = list(workers)
                    async def receive_key(ws, key):
                        async def find():
                            while True:
                                msg = await ws.receive()
                                self.assertEqual(msg.type.name, 'BINARY')
                                if key.encode() in msg.data:
                                    return
                        await asyncio.wait_for(find(), 2)
                    expected = ('background-left' if 'binocular_zmq' in mode else
                                'background-mono' if 'monocular_zmq' in mode else
                                'video' if 'webrtc' in mode else 'motionControllers')
                    await receive_key(a, expected)
                    await receive_key(b, expected)
                    await a.close()
                    await self.wait_for(lambda: workers[ids[0]].done())
                    self.assertIsNone(workers[ids[0]].exception())
                    self.assertFalse(workers[ids[1]].done())
                    self.assertEqual(len(app.ws), 1)
                    if mode != 'main_pass_through':
                        await receive_key(b, expected)
                    await b.close()
                    await self.wait_for(lambda: workers[ids[1]].done() and not app.ws)
                    self.assertIsNone(workers[ids[1]].exception())
                    await asyncio.sleep(.02)  # allow Vuer's uplinks to exit
                    self.assertFalse(errors, str(errors))

    async def test_background_failures_are_diagnosable_and_close_socket(self):
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        errors = []
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        self.addCleanup(loop.set_exception_handler, previous)
        for which in ('producer', 'uplink'):
            with self.subTest(which=which):
                async def producer(session):
                    if which == 'producer':
                        raise RuntimeError('injected producer failure')
                    from vuer.schemas import MotionControllers
                    session.upsert(MotionControllers(key='failure-trigger'), to='bgChildren')
                    while session.CURRENT_WS_ID in session.vuer.ws:
                        await asyncio.sleep(.01)
                app, url = await self.start_server(producer, fail_send=which == 'uplink')
                async with ClientSession() as client:
                    ws = await client.ws_connect(url)
                    await self.wait_for(lambda: not app.ws)
                    await self.wait_for(lambda: any(str(c.get('exception')) == f'injected {which} failure' for c in errors))
                    self.assertEqual([str(c.get('exception')) for c in errors], [f'injected {which} failure'], str(errors))
                    await ws.close()
                errors.clear()


if __name__ == '__main__':
    unittest.main()
