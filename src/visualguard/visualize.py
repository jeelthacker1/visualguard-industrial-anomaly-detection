import numpy as np
from PIL import Image


def overlays(image: Image.Image, anomaly_map: np.ndarray, pixel_threshold: float):
    # Fixed validation-derived scale; never normalize each upload independently.
    scale = max(pixel_threshold * 2.0, 1e-8)
    values = np.clip(anomaly_map / scale, 0, 1)
    # Blue -> cyan -> yellow -> red, with a consistently labeled numerical legend in UI.
    stops = np.array([[25, 45, 115], [29, 181, 186], [249, 220, 80], [219, 50, 37]])
    coordinates = np.linspace(0, 1, len(stops))
    rgb = np.stack(
        [np.interp(values, coordinates, stops[:, channel]) for channel in range(3)], axis=-1
    ).astype(np.uint8)
    heatmap = Image.fromarray(rgb).resize(image.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(image, heatmap, 0.45)
    mask = Image.fromarray(((anomaly_map > pixel_threshold) * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.NEAREST
    )
    return heatmap, overlay, mask
