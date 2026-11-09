from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import VisualGuardError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
PREPROCESSING_VERSION = "rgb-bilinear-imagenet-stretch-v1"


def decode_image(
    data: bytes, *, max_bytes: int = MAX_UPLOAD_BYTES, max_pixels: int = MAX_PIXELS
) -> Image.Image:
    if not data or len(data) > max_bytes:
        raise VisualGuardError(
            f"Image must be nonempty and at most {max_bytes // (1024 * 1024)} MiB."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format not in {"PNG", "JPEG"}:
                    raise VisualGuardError("Only genuine PNG and JPEG image contents are accepted.")
                if probe.width * probe.height > max_pixels or min(probe.size) < 16:
                    raise VisualGuardError(
                        "Image dimensions must be at least 16×16 and at most 16 megapixels."
                    )
                if getattr(probe, "n_frames", 1) != 1:
                    raise VisualGuardError("Animated images are not accepted.")
                if probe.mode not in {"RGB", "L"} or "transparency" in probe.info:
                    raise VisualGuardError(
                        "Use an opaque RGB or grayscale image (no alpha, palette or CMYK)."
                    )
                probe.verify()
            with Image.open(io.BytesIO(data)) as loaded:
                image = ImageOps.exif_transpose(loaded).convert("RGB")
                image.load()
                # Return fresh pixels; discard EXIF, ICC and textual metadata.
                return Image.fromarray(np.asarray(image).copy())
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise VisualGuardError("Invalid, corrupt or excessively large image.") from exc


def read_image(path: Path) -> Image.Image:
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise VisualGuardError("Image exceeds the file size limit.")
    return decode_image(path.read_bytes())


def pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def difference_hash(image: Image.Image) -> str:
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    return f"{int.from_bytes(np.packbits(values[:, 1:] > values[:, :-1]).tobytes(), 'big'):016x}"


def mask_array(
    path: Path, *, original_size: tuple[int, int], size: int | None = None
) -> np.ndarray:
    if path.stat().st_size > MAX_UPLOAD_BYTES or (size is not None and not 1 <= size <= 4096):
        raise VisualGuardError("Mask size exceeds the decoding limit.")
    with Image.open(path) as image:
        if (
            image.size != original_size
            or image.mode not in {"1", "L"}
            or image.width * image.height > MAX_PIXELS
            or image.format != "PNG"
            or image.n_frames != 1
        ):
            raise VisualGuardError(
                "Ground-truth mask dimensions or mode do not match the source image."
            )
        raw = np.asarray(image)
        if not set(np.unique(raw)).issubset({0, 1, 255}):
            raise VisualGuardError("Ground-truth mask is not binary.")
        if size is None:
            return raw > 0
        return np.asarray(image.resize((size, size), Image.Resampling.NEAREST)) > 0


def preprocess(image: Image.Image, size: int):
    import torch

    pixels = (
        np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    )
    pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


@dataclass(frozen=True)
class Quality:
    brightness: float
    contrast: float
    laplacian_variance: float


def image_quality(image: Image.Image) -> Quality:
    from scipy.ndimage import laplace

    gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
    return Quality(float(gray.mean()), float(gray.std()), float(laplace(gray).var()))
