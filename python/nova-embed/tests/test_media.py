"""Per-modality decode / is_empty behavior — the one place transport lives."""

from __future__ import annotations

import io

import pytest

from nova_embed import media
from nova_embed.media import Modality


# ---------------------------------------------------------------- text

def test_text_str_passthrough():
    assert media.decode("hello", Modality.TEXT) == "hello"


def test_text_bytes_decoded():
    assert media.decode("héllo".encode(), Modality.TEXT) == "héllo"


def test_text_wrong_type_raises():
    with pytest.raises(TypeError, match="text modality"):
        media.decode(42, Modality.TEXT)


@pytest.mark.parametrize(
    ("value", "empty"),
    [
        (None, True),
        ("", True),
        ("   \n\t", True),
        (b"", True),
        ("x", False),
        (b"x", False),
    ],
)
def test_text_is_empty(value, empty):
    assert media.is_empty(value, Modality.TEXT) is empty


# ---------------------------------------------------------------- image
#
# Everything below needs pillow (nova-embed[embed]); each test starts with the
# importorskip via _png_bytes or its own guard.


def _png_bytes() -> bytes:
    pytest.importorskip("PIL")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_image_from_bytes():
    pytest.importorskip("PIL")
    from PIL import Image

    img = media.decode(_png_bytes(), Modality.IMAGE)
    assert isinstance(img, Image.Image)
    assert img.size == (2, 2)


def test_image_from_path(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    p = tmp_path / "img.png"
    p.write_bytes(_png_bytes())
    img = media.decode(str(p), Modality.IMAGE)
    assert isinstance(img, Image.Image)


def test_image_from_hf_dict_prefers_bytes(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    img = media.decode({"bytes": _png_bytes(), "path": "/nonexistent"}, Modality.IMAGE)
    assert isinstance(img, Image.Image)


def test_image_pil_passthrough():
    pytest.importorskip("PIL")
    from PIL import Image

    original = Image.new("RGB", (1, 1))
    assert media.decode(original, Modality.IMAGE) is original


@pytest.mark.parametrize(
    ("value", "empty"),
    [
        (None, True),
        (b"", True),
        ("", True),
        ({"bytes": None, "path": None}, True),
        ({"bytes": b"x", "path": None}, False),
        ({"bytes": None, "path": "/some/img.png"}, False),
        (b"x", False),
        ("path.png", False),
    ],
)
def test_image_is_empty(value, empty):
    assert media.is_empty(value, Modality.IMAGE) is empty


def test_image_empty_dict_decode_raises():
    with pytest.raises(TypeError, match="neither"):
        media.decode({"bytes": None, "path": None}, Modality.IMAGE)
