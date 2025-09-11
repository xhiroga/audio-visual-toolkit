from audio_visual_toolkit.validate_label import validate_label
from pathlib import Path


def test_validate_label_true() -> None:
    rel = Path(__file__).parent / "assets" / "rct001.lab"
    lab_path = rel.resolve()

    assert validate_label(str(lab_path)) is True


def test_validate_label_false() -> None:
    rel = Path(__file__).parent / "assets" / "wrong.lab"
    lab_path = rel.resolve()

    assert validate_label(str(lab_path)) is False
