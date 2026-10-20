import ast
import inspect
from pathlib import Path


SRC = Path(__file__).parents[1] / "src" / "televuer"


def _init_default(path, class_name, argument):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    names = [arg.arg for arg in init.args.args]
    defaults = [None] * (len(names) - len(init.args.defaults)) + init.args.defaults
    return ast.literal_eval(defaults[names.index(argument)])


def test_jpeg_quality_defaults_to_legacy_80_in_core_and_wrapper():
    assert _init_default(SRC / "televuer.py", "TeleVuer", "jpeg_quality") == 80
    assert _init_default(SRC / "tv_wrapper.py", "TeleVuerWrapper", "jpeg_quality") == 80


def _init_arg_names(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    return [arg.arg for arg in init.args.args]


def test_jpeg_quality_is_appended_without_shifting_legacy_positional_arguments():
    for filename, class_name in (("televuer.py", "TeleVuer"), ("tv_wrapper.py", "TeleVuerWrapper")):
        names = _init_arg_names(SRC / filename, class_name)
        assert names[5] == "display_mode"
        assert names[-1] == "jpeg_quality"


def test_public_constructor_signatures_bind_legacy_positional_arguments_unchanged():
    from televuer.televuer import TeleVuer
    from televuer.tv_wrapper import TeleVuerWrapper

    legacy_values = [False, True, (480, 1280), 12.0, "ego", True, False, None]
    for constructor in (TeleVuer, TeleVuerWrapper):
        bound = inspect.signature(constructor).bind(*legacy_values)
        assert bound.arguments["display_mode"] == "ego"
        assert bound.arguments["zmq"] is True
        assert bound.arguments["webrtc"] is False
        assert bound.arguments["webrtc_url"] is None


def test_all_zmq_image_backgrounds_use_configured_jpeg_quality():
    source = (SRC / "televuer.py").read_text(encoding="utf-8")

    assert "self.jpeg_quality = jpeg_quality" in source
    assert "quality=80" not in source
    assert source.count("quality=self.jpeg_quality") == 6


def test_wrapper_forwards_jpeg_quality_to_televuer():
    source = (SRC / "tv_wrapper.py").read_text(encoding="utf-8")
    assert "jpeg_quality=jpeg_quality" in source