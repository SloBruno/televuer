"""Bounded, best-effort video on Vuer's control WebSocket.

Only image encoding runs off-loop. Vuer session/queue mutation remains on the
owning event loop. Never clear that queue: it also carries scene and haptics.
This bounds SERVER backlog; it is not an acknowledgement of headset rendering.
"""
import asyncio
import time

from vuer import Vuer


class VideoVuer(Vuer):
    """Treat a lost peer during a blocked send as normal disconnect.

    Pinned Vuer handles ConnectionResetError but re-raises its base
    ConnectionError after cleanup. Do not hide encoding/application errors.
    """
    async def uplink(self, proxy):
        try:
            return await super().uplink(proxy)
        except ConnectionError:
            await self.close_ws(proxy.CURRENT_WS_ID)


class VideoFrameSender:
    def __init__(self, session, max_buffer_bytes=65536):
        self.session = session
        self.max_buffer_bytes = max_buffer_bytes
        self.stats = dict(enqueued=0, skipped=0, encode_ms=0.0,
                          buffer_bytes=0, max_buffer_bytes=0)

    def connected(self):
        try:
            return not self.session.socket.closed
        except KeyError:
            # Vuer removes the websocket from its pool on disconnect.
            return False

    def ready(self):
        if not self.connected():
            return False
        # The pinned Vuer 0.0.60 session API. Defer video if ANY message is
        # pending, instead of replacing/dropping controller scene setup/haptics.
        if self.session.uplink_queue:
            return False
        transport = self.session.socket._writer.transport
        size = transport.get_write_buffer_size()
        self.stats['buffer_bytes'] = size
        self.stats['max_buffer_bytes'] = max(size, self.stats['max_buffer_bytes'])
        return size < self.max_buffer_bytes

    async def publish(self, build_elements, *, to='bgChildren'):
        if not self.ready():
            self.stats['skipped'] += 1
            return False
        start = time.monotonic()
        # Await ONE encode before accepting the next frame: no executor FIFO
        # of obsolete images. The factory must snapshot the shared image.
        elements = await asyncio.to_thread(build_elements)
        self.stats['encode_ms'] = (time.monotonic() - start) * 1000
        # Disconnect or congestion may have happened while JPEG was encoding.
        if not self.ready():
            self.stats['skipped'] += 1
            return False
        self.session.upsert(elements, to=to)
        self.stats['enqueued'] += 1
        return True
