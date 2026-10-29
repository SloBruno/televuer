"""Vuer 0.0.60 session-lifetime utilities."""


def install_socket_lifecycle(vuer):
    """Fix Vuer 0.0.60's websocket response and exceptional cleanup locally."""
    import logging
    from aiohttp import web

    original = vuer.socket_index

    async def socket_index(request):
        if request.headers.get('Upgrade', '').lower().strip() != 'websocket':
            return await original(request)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            await vuer.downlink(request, ws)
        except Exception:
            logging.getLogger(__name__).exception('Vuer websocket downlink failed')
        finally:
            # Vuer's exception branch raises before close_ws, leaving the ID.
            for ws_id, candidate in list(vuer.ws.items()):
                if candidate is ws:
                    await vuer.close_ws(ws_id)
            await ws.close()
        return ws

    vuer.socket_index = socket_index


def session_is_open(session):
    """Check the session's own socket, not another connected client's socket.

    Vuer removes closed sockets before its spawned producer terminates.  This
    check and the ensuing synchronous upsert run on the same asyncio loop.
    """
    socket = session.vuer.ws.get(session.CURRENT_WS_ID)
    return socket is not None and not socket.closed
