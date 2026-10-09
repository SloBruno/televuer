import importlib
import signal
import sys
import types

from test_head_tracking import _import_televuer


def test_vuer_child_restores_default_sigterm_before_server_run(monkeypatch):
    module = _import_televuer(monkeypatch)
    viewer = module.TeleVuer.__new__(module.TeleVuer)
    calls = []
    viewer.vuer = types.SimpleNamespace(run=lambda: calls.append("run"))
    monkeypatch.setattr(module.signal, "signal", lambda sig, handler: calls.append((sig, handler)))

    viewer._vuer_run()

    assert calls[0] == (signal.SIGTERM, signal.SIG_DFL)
    assert calls[1] == "run"


def test_close_escalates_to_kill_when_child_ignores_terminate(monkeypatch):
    module = _import_televuer(monkeypatch)
    viewer = module.TeleVuer.__new__(module.TeleVuer)

    class StubbornProcess:
        def __init__(self):
            self.calls = []
            self.alive = True

        def terminate(self):
            self.calls.append("terminate")

        def join(self, timeout=None):
            self.calls.append(("join", timeout))
            if self.calls.count("kill"):
                self.alive = False

        def is_alive(self):
            return self.alive

        def kill(self):
            self.calls.append("kill")

    viewer.process = StubbornProcess()
    viewer.display_mode = "pass-through"

    viewer.close()

    assert viewer.process.calls == [
        "terminate", ("join", 0.5), "kill", ("join", 0.5)
    ]
    assert viewer.process.is_alive() is False


def test_close_is_idempotent_after_child_has_exited(monkeypatch):
    module = _import_televuer(monkeypatch)
    viewer = module.TeleVuer.__new__(module.TeleVuer)

    class ExitedProcess:
        def __init__(self):
            self.calls = []

        def is_alive(self):
            return False

        def close(self):
            self.calls.append("close")

    viewer.process = ExitedProcess()
    viewer.display_mode = "pass-through"

    viewer.close()
    viewer.close()

    assert viewer.process.calls == ["close"]
