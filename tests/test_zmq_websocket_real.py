"""Optional real teleimager integration; set TELEIMAGER_SRC to its src directory."""
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_real_zmq_websocket_pipeline():
    source = os.environ.get('TELEIMAGER_SRC')
    if not source:
        pytest.skip('set TELEIMAGER_SRC to the actual teleimager/src directory')
    assert (Path(source) / 'teleimager' / 'image_client.py').is_file()
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'check_zmq_websocket_real.py'
    subprocess.run([sys.executable, str(script)], check=True, timeout=25)
