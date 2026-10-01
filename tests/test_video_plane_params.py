import re
from pathlib import Path
SRC = Path(__file__).parents[1] / "src" / "televuer"

def test_defaults_preserve_historical_plane():
    t = (SRC / "televuer.py").read_text(); w = (SRC / "tv_wrapper.py").read_text()
    for s in (t, w):
        assert "video_plane_height: float=1.0" in s and "video_plane_distance: float=1.0" in s
    assert "height=self.video_plane_height" in t and "distanceToCamera=self.video_plane_distance" in t
    assert "video_plane_height=video_plane_height" in w
