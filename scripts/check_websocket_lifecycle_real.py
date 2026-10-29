"""Loopback integration test: real Vuer 0.0.60 and aiohttp WebSockets; no robot."""
import asyncio
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
from aiohttp import ClientSession, web
from vuer import Vuer

class WebsocketLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_reconnect_cleans_session_workers(self):
        errors = []
        loop = asyncio.get_running_loop()
        old_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        app = Vuer(host='127.0.0.1', queue_len=3)
        workers = []
        viewer = TeleVuer.__new__(TeleVuer)
        viewer.use_hand_tracking = False
        viewer.display_fps = 50
        viewer.webrtc_url = 'http://127.0.0.1/video'
        viewer.aspect_ratio = 1.0
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
                for _ in range(2):
                    async with client.ws_connect(f'http://127.0.0.1:{port}/') as ws:
                        msg = await asyncio.wait_for(ws.receive(), 2)
                        self.assertEqual(msg.type.name, 'BINARY')
                    await asyncio.sleep(.1)
                    self.assertTrue(workers[-1].done(), 'worker still alive after disconnect')
                    self.assertIsNone(workers[-1].exception(), 'worker failed after disconnect')
                    self.assertFalse(app.ws)
            self.assertFalse(errors, str(errors))
        finally:
            await runner.cleanup()
            loop.set_exception_handler(old_handler)

if __name__ == '__main__':
    unittest.main()
