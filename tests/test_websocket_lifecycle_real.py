"""Run real WebSocket lifecycle integration in a clean interpreter."""
import subprocess
import sys
from pathlib import Path


def test_websocket_disconnect_reconnect():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'check_websocket_lifecycle_real.py'
    subprocess.run([sys.executable, str(script)], check=True, timeout=20)


def test_websocket_modes_and_failures():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'check_websocket_modes_real.py'
    subprocess.run([sys.executable, str(script)], check=True, timeout=30)
