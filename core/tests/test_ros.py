import pytest
from core.data.ros import ColorRGBA


def test_color_rgba_from_hex_6():
    color = ColorRGBA.from_hex("#FF0000")
    assert color.r == 1.0
    assert color.g == 0.0
    assert color.b == 0.0
    assert color.a == 1.0

    color = ColorRGBA.from_hex("00FF00")
    assert color.r == 0.0
    assert color.g == 1.0
    assert color.b == 0.0
    assert color.a == 1.0


def test_color_rgba_from_hex_8():
    color = ColorRGBA.from_hex("#0000FF7F")
    assert color.r == 0.0
    assert color.g == 0.0
    assert color.b == 1.0
    assert pytest.approx(color.a, abs=0.01) == 0.5


def test_color_rgba_from_hex_invalid():
    with pytest.raises(ValueError):
        ColorRGBA.from_hex("#ABC")
    with pytest.raises(ValueError):
        ColorRGBA.from_hex("#GG0000")
