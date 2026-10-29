"""Loopback-only integration with a real external teleimager ImageClient.

Run with TELEIMAGER_SRC pointing to the teleimager/src directory. The package is
copied to a temporary directory because ImageClient persists camera config.
"""
import asyncio
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

import cv2
import msgpack
import numpy as np
import zmq
from aiohttp import ClientSession, web
from vuer import Vuer

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
haptics = types.ModuleType('teleop.utils.haptics')
haptics.HapticTransportAdapter = object
sys.modules['teleop'] = types.ModuleType('teleop')
sys.modules['teleop.utils'] = types.ModuleType('teleop.utils')
sys.modules['teleop.utils.haptics'] = haptics
from televuer.televuer import TeleVuer
from televuer.session_lifecycle import install_socket_lifecycle


def images(obj):
    if isinstance(obj, dict):
        if obj.get('key') == 'background-mono' and isinstance(obj.get('src'), bytes):
            yield obj['src']
        for v in obj.values():
            yield from images(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from images(v)


class Pipeline(unittest.IsolatedAsyncioTestCase):
    async def test_real_pipeline_reconnect_and_malformed_message(self):
        errors = []
        loop = asyncio.get_running_loop()
        old = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        self.addCleanup(loop.set_exception_handler, old)
        class Capture(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    errors.append(self.format(record))
        capture = Capture()
        loggers = [logging.getLogger('televuer.session_lifecycle'), logging.getLogger('aiohttp.server')]
        for logger in loggers:
            logger.addHandler(capture)
            self.addCleanup(logger.removeHandler, capture)
        thread_errors = []
        old_hook = threading.excepthook
        threading.excepthook = lambda args: thread_errors.append(args)
        self.addCleanup(setattr, threading, 'excepthook', old_hook)

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'package' / 'src'
            shutil.copytree(Path(os.environ['TELEIMAGER_SRC']) / 'teleimager', source / 'teleimager')
            sys.path.insert(0, str(source))
            self.addCleanup(sys.path.remove, str(source))
            from teleimager.image_client import ImageClient
            context = zmq.Context()
            pub = context.socket(zmq.PUB)
            pub.setsockopt(zmq.LINGER, 0)
            pub_port = pub.bind_to_random_port('tcp://127.0.0.1')
            rep = context.socket(zmq.REP)
            rep.setsockopt(zmq.LINGER, 0)
            rep_port = rep.bind_to_random_port('tcp://127.0.0.1')
            config = {'head_camera': {'enable_zmq': True, 'enable_webrtc': False,
                                      'zmq_port': pub_port, 'binocular': False,
                                      'image_shape': [32, 32], 'fps': 30},
                      'left_wrist_camera': {'enable_zmq': False},
                      'right_wrist_camera': {'enable_zmq': False}}
            def respond():
                try:
                    if rep.poll(4000):
                        rep.recv()
                        rep.send_json(config)
                    else:
                        raise TimeoutError('config request not received')
                except Exception as exc:
                    thread_errors.append(exc)
            responder = threading.Thread(target=respond)
            responder.start()
            client = None
            viewer = TeleVuer.__new__(TeleVuer)
            viewer.use_hand_tracking = False
            viewer.display_fps = 30
            viewer.display_mode = 'immersive'
            viewer.webrtc = False
            viewer.aspect_ratio = viewer.video_plane_height = viewer.video_plane_distance = 1.0
            viewer.img2display = np.zeros((32, 32, 3), dtype=np.uint8)
            viewer.latest_frame = None
            viewer.new_frame_event = threading.Event()
            viewer.stop_writer_event = threading.Event()
            viewer.writer_thread = threading.Thread(target=viewer._xr_render_loop)
            viewer.writer_thread.start()
            app = Vuer(host='127.0.0.1', queue_len=3)
            install_socket_lifecycle(app)
            workers = []
            async def producer(session):
                workers.append(asyncio.current_task())
                await viewer.main_image_monocular_zmq(session)
            app.spawn(start=False)(producer)
            app._add_route('/', app.socket_index)
            runner = web.AppRunner(app.app)
            await runner.setup()
            site = web.TCPSite(runner, '127.0.0.1', 0)
            await site.start()
            url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/'
            async def wait_for(predicate, timeout=4):
                async def wait():
                    while not predicate():
                        await asyncio.sleep(.01)
                await asyncio.wait_for(wait(), timeout)
            try:
                client = await asyncio.to_thread(ImageClient, host='127.0.0.1', request_port=rep_port, request_bgr=True, camera_names=['head_camera'])
                self.assertEqual(client.get_cam_config(), config)
                async def transmit(bgr):
                    ok, jpg = cv2.imencode('.jpg', np.full((32, 32, 3), bgr, dtype=np.uint8))
                    self.assertTrue(ok)
                    async def poll():
                        while True:
                            pub.send(jpg.tobytes())
                            frame = client.get_head_frame().bgr
                            if frame is not None and np.max(np.abs(frame.astype(int).mean((0, 1)) - bgr)) < 5:
                                viewer.render_to_xr(frame)
                                return
                            await asyncio.sleep(.03)
                    await asyncio.wait_for(poll(), 4)
                async def receive_image(ws, expected_rgb):
                    async def receive():
                        while True:
                            msg = await ws.receive()
                            self.assertEqual(msg.type.name, 'BINARY', repr(msg))
                            for data in images(msgpack.unpackb(msg.data, raw=False)):
                                decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                                if decoded is not None:
                                    rgb = decoded[:, :, ::-1].mean((0, 1))
                                    if np.max(np.abs(rgb - expected_rgb)) < 8:
                                        return
                    await asyncio.wait_for(receive(), 4)
                async with ClientSession() as browser:
                    for cycle, bgr in enumerate(([20, 80, 200], [190, 50, 30], [30, 180, 90])):
                        await transmit(bgr)
                        async with browser.ws_connect(url) as ws:
                            await receive_image(ws, list(reversed(bgr)))
                            if cycle == 1:
                                ws._response.connection.transport.abort()
                            if cycle == 2:
                                await ws.send_bytes(b'\xc1')
                                await wait_for(lambda: not app.ws)
                        await wait_for(lambda: len(workers) == cycle + 1 and workers[-1].done() and not app.ws)
                        self.assertIsNone(workers[-1].exception())
                await asyncio.sleep(.1)
                self.assertEqual(len(errors), 1, str(errors))
                self.assertIn('FormatError', str(errors[0]))
                self.assertFalse(thread_errors, str(thread_errors))
            finally:
                await runner.cleanup()
                viewer.stop_writer_event.set()
                viewer.new_frame_event.set()
                viewer.writer_thread.join(2)
                if client:
                    client.close()
                    client._requester.close()
                responder.join(5)
                pub.close()
                rep.close()
                context.term()
                self.assertFalse(viewer.writer_thread.is_alive())
                self.assertFalse(responder.is_alive())
                self.assertFalse(thread_errors, str(thread_errors))

if __name__ == '__main__':
    unittest.main()
