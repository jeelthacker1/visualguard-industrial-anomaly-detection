"""Own-created fixtures for engineering tests; not industrial accuracy evidence."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .errors import VisualGuardError
from .utils import write_json


def generate(root: Path, *, seed: int = 123, count: int = 20) -> None:
    if root.exists():
        raise VisualGuardError("Synthetic output must be a new directory.")
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    for source, defect, number in [
        ("train", "good", count),
        ("test", "good", 5),
        ("test", "scratch", 5),
    ]:
        folder = root / "synthetic_part" / source / defect
        folder.mkdir(parents=True)
        for index in range(number):
            array = np.clip(rng.normal(50, 5, (64, 64, 3)), 0, 255).astype(np.uint8)
            image = Image.fromarray(array)
            draw = ImageDraw.Draw(image)
            offset = int(rng.integers(-2, 3))
            draw.rounded_rectangle(
                (14 + offset, 10, 50 + offset, 54), radius=5, fill=(150 + index % 15, 158, 170)
            )
            mask = Image.new("L", image.size, 0)
            if defect != "good":
                draw.line((22, 20 + index, 44, 39 + index), fill=(250, 245, 230), width=4)
                ImageDraw.Draw(mask).line((22, 20 + index, 44, 39 + index), fill=255, width=4)
                mask_path = root / "synthetic_part" / "ground_truth" / defect
                mask_path.mkdir(parents=True, exist_ok=True)
                mask.save(mask_path / f"{index:03d}_mask.png")
            image.save(folder / f"{index:03d}.png")
    write_json(
        root / "synthetic_provenance.json",
        {
            "generator": "visualguard-own-fixtures-v1",
            "seed": seed,
            "purpose": "software smoke testing only; no industrial accuracy claims",
        },
    )
