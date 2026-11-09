"""Real Vuer/aiohttp transport, loopback-only, no robot or DDS imports."""
import asyncio
import cv2
import socket
import time

from aiohttp import web, ClientSession
from msgpack import packb, unpackb
import numpy as np
from televuer.video_sender import VideoVuer as Vuer

from televuer.televuer import TeleVuer


def test_real_vuer_slow_consumer_control_and_reconnect():
    async def run():
        app = Vuer(host='127.0.0.1', queue_len=3)
        tasks, errors, viewers, receipts = [], [], [], []
        def add_task(coro, name=None):
            task = asyncio.create_task(coro, name=name)
            tasks.append(task)
            def done(t):
                if not t.cancelled() and t.exception() is not None:
                    errors.append(repr(t.exception()))
            task.add_done_callback(done)
            return task
        app._add_task = add_task
        app._socket('/ws', app.downlink)
        async def on_probe(event, session):
            receipts.append(event.value)
        app.add_handler('PASSIVE_TEST', on_probe)

        @app.spawn(start=False)
        async def video(session):
            # Force real TCP backpressure without sending huge volumes.
            transport = session.socket._writer.transport
            transport.get_extra_info('socket').setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32768)
            tv = TeleVuer.__new__(TeleVuer)
            tv.use_hand_tracking = False
            tv.img2display = np.random.default_rng(42).integers(0, 256, (480, 1280, 3), dtype=np.uint8)
            tv.img_width, tv.aspect_ratio = 640, 4 / 3
            tv.display_fps, tv.jpeg_quality = 100, 60
            viewers.append(tv)
            await tv.main_image_binocular_zmq(session)

        runner = web.AppRunner(app.app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        url = f'http://127.0.0.1:{port}/ws'
        try:
            async with ClientSession() as client:
                ws = await client.ws_connect(url, max_msg_size=2**23)
                # Stop reading video, but keep sending input on the same socket.
                peer_transport = ws._response.connection.transport
                peer_transport.pause_reading()
                for i in range(50):
                    await ws.send_bytes(packb(dict(etype='PASSIVE_TEST', value=i)))
                    await asyncio.sleep(0.02)
                assert receipts == list(range(50)), receipts
                stats = viewers[0].video_stream_stats
                assert stats['skipped'] > 0, stats
                assert stats['enqueued'] < 30, stats
                print('slow_consumer', stats, 'input_events', len(receipts))
                # Abrupt transport loss must not leave a scene producer throwing.
                peer_transport.abort()
                await asyncio.sleep(0.2)

                ws2 = await client.ws_connect(url, max_msg_size=2**23)
                found = False
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    message = await asyncio.wait_for(ws2.receive(), 1)
                    if message.type.name != 'BINARY':
                        continue
                    payload = unpackb(message.data, raw=False)
                    if payload.get('etype') == 'UPSERT':
                        children = payload.get('data', {}).get('nodes', [])
                        keys = {x.get('key') for x in children}
                        if keys == {'background-left', 'background-right'}:
                            assert all(isinstance(x['src'], bytes) for x in children)
                            for child in children:
                                image = cv2.imdecode(np.frombuffer(child['src'], np.uint8), cv2.IMREAD_COLOR)
                                assert image.shape == (480, 640, 3)
                            found = True
                            break
                assert found, payload
                await ws2.close()
                await asyncio.sleep(0.2)
        finally:
            # Include background handler errors, not just client exit status.
            for ws in list(app.ws.values()):
                await ws.close()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await runner.cleanup()
        assert errors == [], errors
    asyncio.run(run())
