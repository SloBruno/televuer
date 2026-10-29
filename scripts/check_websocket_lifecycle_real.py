"""Loopback integration test: real Vuer 0.0.60 and aiohttp WebSockets; no robot."""
import asyncio
import logging
from multiprocessing import Array
import msgpack
import unittest
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.modules['cv2'] = types.ModuleType('cv2')
package = types.ModuleType('televuer')
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'src' / 'televuer')]
sys.modules['televuer'] = package
haptics = types.ModuleType('teleop.utils.haptics')
haptics.HapticTransportAdapter = object
sys.modules['teleop'] = types.ModuleType('teleop')
sys.modules['teleop.utils'] = types.ModuleType('teleop.utils')
sys.modules['teleop.utils.haptics'] = haptics
from televuer.televuer import TeleVuer
from televuer.session_lifecycle import install_socket_lifecycle
from aiohttp import ClientSession, web
from vuer import Vuer

class WebsocketLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_reconnect_cleans_session_workers(self):
        errors = []
        class ErrorCollector(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    errors.append(self.format(record))
        collector = ErrorCollector()
        logger = logging.getLogger('aiohttp.server')
        logger.addHandler(collector)
        lifecycle_logger = logging.getLogger('televuer.session_lifecycle')
        lifecycle_logger.addHandler(collector)
        loop = asyncio.get_running_loop()
        old_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        app = Vuer(host='127.0.0.1', queue_len=3)
        install_socket_lifecycle(app)
        workers = []
        viewer = TeleVuer.__new__(TeleVuer)
        viewer.use_hand_tracking = False
        viewer.display_fps = 50
        viewer.webrtc_url = 'http://127.0.0.1/video'
        viewer.aspect_ratio = 1.0
        viewer.client_info_shared = Array('c', 512, lock=True)
        app.add_handler('INIT')(viewer.on_client_init)
        async def producer(session):
            workers.append(asyncio.current_task())
            await viewer.main_image_monocular_webrtc(session)
        app.spawn(start=False)(producer)
        app._add_route('/', app.socket_index)
        runner = web.AppRunner(app.app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with ClientSession() as client:
                for cycle in range(5):
                    async with client.ws_connect(f'http://127.0.0.1:{port}/') as ws:
                        msg = await asyncio.wait_for(ws.receive(), 2)
                        self.assertEqual(msg.type.name, 'BINARY')
                        if cycle in (0, 2, 4):
                            modern = cycle != 2
                            value = {'client': 'browser', 'pkg': '@vuer-ai/vuer',
                                     'pkgVersion': '0.0.103' if modern else '0.0.60',
                                     'userAgent': 'QuestBrowser' if cycle == 0 else 'PicoBrowser'}
                            await ws.send_bytes(msgpack.packb({'etype': 'INIT', 'value': value}))
                            for _ in range(100):
                                if viewer.client_info.get('userAgent') == value['userAgent'] and viewer.client_info.get('pkgVersion') == value['pkgVersion']:
                                    break
                                await asyncio.sleep(.01)
                            self.assertEqual(viewer.client_info.get('headComponentExpected'), modern)
                        if cycle == 1:
                            # Break the transport without a WebSocket close frame.
                            ws._response.connection.transport.abort()
                        if cycle == 3:
                            await ws.send_bytes(b'\xc1')  # invalid MessagePack
                            await asyncio.sleep(.05)
                    await asyncio.sleep(.1)
                    self.assertTrue(workers[-1].done(), 'worker still alive after disconnect')
                    self.assertIsNone(workers[-1].exception(), 'worker failed after disconnect')
                    self.assertFalse(app.ws)
            self.assertEqual(len(errors), 1, str(errors))
            self.assertIn('Vuer websocket downlink failed', errors[0])
            self.assertIn('FormatError', errors[0])
        finally:
            await runner.cleanup()
            loop.set_exception_handler(old_handler)
            logger.removeHandler(collector)
            lifecycle_logger.removeHandler(collector)

if __name__ == '__main__':
    unittest.main()
