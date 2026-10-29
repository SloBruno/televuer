"""Vuer 0.0.60 session-lifetime utilities."""


def session_is_open(session):
    """Check the session's own socket, not another connected client's socket.

    Vuer removes closed sockets before its spawned producer terminates.  This
    check and the ensuing synchronous upsert run on the same asyncio loop.
    """
    socket = session.vuer.ws.get(session.CURRENT_WS_ID)
    return socket is not None and not socket.closed
