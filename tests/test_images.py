import io

import numpy as np
import pytest
from PIL import Image

from visualguard.errors import VisualGuardError
from visualguard.images import MAX_UPLOAD_BYTES, decode_image, mask_array


def test_decode_valid_png(png):
    image = decode_image(png)
    assert image.size == (64, 64) and image.mode == "RGB"
    assert not image.info


@pytest.mark.parametrize(
    "data", [b"", b"not an image", b"<svg onload=alert(1)></svg>", b"\x89PNG\r\n\x1a\ncorrupt"]
)
def test_corrupt_upload_rejected(data):
    with pytest.raises(VisualGuardError):
        decode_image(data)


def test_byte_limit():
    with pytest.raises(VisualGuardError):
        decode_image(b"0" * (MAX_UPLOAD_BYTES + 1))


def test_png_transparency_in_rgb_metadata():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, "PNG", transparency=(0, 0, 0))
    with pytest.raises(VisualGuardError, match="opaque"):
        decode_image(buffer.getvalue())


def test_wide_mask_audit_preserves_original_geometry(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("L", (16000, 16), 255).save(path)
    assert mask_array(path, original_size=(16000, 16)).shape == (16, 16000)


def test_pixel_limit_before_decoding(png):
    with pytest.raises(VisualGuardError, match="dimensions"):
        decode_image(png, max_pixels=100)


@pytest.mark.parametrize(
    "mode,fmt", [("RGBA", "PNG"), ("P", "PNG"), ("RGB", "GIF"), ("CMYK", "JPEG")]
)
def test_ambiguous_or_unsupported_modes(mode, fmt):
    buffer = io.BytesIO()
    Image.new(mode, (32, 32)).save(buffer, format=fmt)
    with pytest.raises(VisualGuardError):
        decode_image(buffer.getvalue())


def test_nearest_mask_resize(dataset):
    root, samples = dataset
    row = next(s for s in samples if s.mask)
    mask = mask_array(root / row.mask, original_size=(64, 64), size=27)
    assert mask.shape == (27, 27)
    assert set(np.unique(mask)) == {False, True}
    with pytest.raises(VisualGuardError):
        mask_array(root / row.mask, original_size=(32, 32), size=27)
