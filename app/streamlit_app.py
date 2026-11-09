from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from pathlib import Path

import streamlit as st

from visualguard.artifacts import load_artifact
from visualguard.errors import VisualGuardError
from visualguard.images import MAX_UPLOAD_BYTES
from visualguard.inference import inspect_bytes
from visualguard.models import configure_torch
from visualguard.visualize import overlays

st.set_page_config(page_title="VisualGuard · Inspection lab", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.stApp {background:#f4f5f1;color:#132d37}
[data-testid=stSidebar] {background:#e7ece8}
[data-testid=stMetric] {background:white;padding:16px;border:1px solid #d4deda;border-radius:12px}
h1,h2,h3 {letter-spacing:-.035em}
.vg-eyebrow {font-size:12px;letter-spacing:.22em;color:#52736a;font-weight:700}
.vg-hero {padding:20px 0 12px;border-bottom:1px solid #cad6d0;margin-bottom:24px}
.vg-hero h1 {font-size:48px;margin:8px 0}.vg-hero p {color:#58706b;max-width:800px}
</style><div class="vg-hero"><div class="vg-eyebrow">VISUALGUARD / MACHINE VISION RESEARCH</div>
<h1>Look closer. Inspect with evidence.</h1><p>Compare a part with learned normal appearances. Review anomaly scores, localize unusual regions, and inspect the evidence behind every result.</p></div>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def resources(path: str, integrity_hash: str):
    detector, meta = load_artifact(Path(path))
    configure_torch(detector.config)
    return detector, meta, threading.Lock()


def discover_models(root):
    found = []
    if root.is_dir():
        for meta_path in sorted(root.glob("**/metadata.json")):
            if not meta_path.resolve().is_relative_to(root.resolve()) or meta_path.is_symlink():
                continue
            try:
                if meta_path.stat().st_size > 2 * 1024**2:
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = f"{meta['provenance']['category']} · {meta['config']['method']} · {meta['model_id'][:8]}"
                found.append((name, meta_path.parent, meta))
            except (OSError, ValueError, KeyError):
                continue
    return found


artifact_root = Path(os.environ.get("VISUALGUARD_MODEL_DIR", "artifacts"))
models = discover_models(artifact_root)
with st.sidebar:
    st.markdown("### Inspection settings")
    st.caption("Each reference bank is specific to one product category.")
    selection = (
        st.selectbox("Trained model", range(len(models)), format_func=lambda i: models[i][0])
        if models
        else None
    )
    deadline = st.number_input(
        "Review if processing exceeds (ms)", min_value=1, max_value=60000, value=1000
    )
    st.divider()
    st.markdown("**Research prototype**")
    st.caption(
        "An anomaly score is not a probability or safety certificate. Camera changes, lighting, reflections and tiny defects can change results."
    )
    st.caption("Uploads are processed in memory. The app does not write them to disk.")

inspect_tab, evidence_tab, guide_tab = st.tabs(["Inspect a part", "Model evidence", "How it works"])
with inspect_tab:
    if not models:
        st.info(
            "No trained model is installed yet. Prepare the dataset and train a category from the command line."
        )
        st.code(
            "visualguard train --root local_data/MVTecAD --manifest splits/bottle_seed42.csv --config configs/patchcore_resnet18.yaml --output artifacts/bottle_patchcore"
        )
    else:
        _, model_path, selected_meta = models[selection]
        if selected_meta["provenance"]["dataset"] == "synthetic-smoke":
            st.warning(
                "SOFTWARE DEMO ONLY · This model uses synthetic fixtures and random weights. It does not demonstrate industrial accuracy."
            )
        file = st.file_uploader(
            "Upload a part from the selected category",
            type=["png", "jpg", "jpeg"],
            help="Opaque RGB/grayscale PNG or JPEG. Maximum 10 MiB and 16 megapixels.",
        )
        st.caption(
            "Wrong product categories are not automatically identified. Confirm the chosen category before inspection."
        )
        if file is not None and st.button("Inspect image", type="primary"):
            try:
                if file.size > MAX_UPLOAD_BYTES:
                    raise VisualGuardError("Upload exceeds 10 MiB.")
                integrity = hashlib.sha256((model_path / "integrity.json").read_bytes()).hexdigest()
                detector, meta, lock = resources(str(model_path), integrity)
                if not lock.acquire(blocking=False):
                    raise VisualGuardError(
                        "The shared inspector is busy. Try again when the current inspection finishes."
                    )
                try:
                    with st.spinner("Comparing with the normal reference bank…"):
                        result, image, anomaly_map = inspect_bytes(
                            file.getvalue(), detector, meta, deadline_ms=deadline
                        )
                finally:
                    lock.release()
                if result["decision"] == "review_required":
                    st.warning("Review required · anomaly threshold or acquisition check triggered")
                else:
                    st.success("Normal appearance · within the learned reference threshold")
                a, b, c, d = st.columns(4)
                a.metric("Anomaly score", f"{result['anomaly_score']:.4f}")
                b.metric("Fixed threshold", f"{result['threshold']:.4f}")
                c.metric("Inspection time", f"{result['latency_ms']:.1f} ms")
                d.metric("Category", result["category"])
                for message in result["warnings"]:
                    st.caption(f"Review: {message}")
                if anomaly_map is not None:
                    heatmap, overlay, mask = overlays(image, anomaly_map, result["pixel_threshold"])
                    columns = st.columns(4)
                    for column, picture, caption in zip(
                        columns,
                        [image, heatmap, overlay, mask],
                        [
                            "Original",
                            "Continuous anomaly map",
                            "Overlay",
                            "Validation-threshold mask",
                        ],
                        strict=True,
                    ):
                        column.image(picture, caption=caption, width="stretch")
                    st.caption(
                        f"Heatmap scale: blue = 0; red ≥ {2 * result['pixel_threshold']:.4f}. Fixed pixel threshold = {result['pixel_threshold']:.4f}. Overlay is mapped back to the original dimensions."
                    )
                    buffer = io.BytesIO()
                    overlay.save(buffer, format="PNG")
                    st.download_button(
                        "Download overlay",
                        buffer.getvalue(),
                        "visualguard_overlay.png",
                        "image/png",
                        on_click="ignore",
                    )
                else:
                    st.image(
                        image,
                        caption="Global embedding baseline: image-level score only",
                        width=450,
                    )
                st.download_button(
                    "Download inspection JSON",
                    json.dumps(result, indent=2),
                    "visualguard_inspection.json",
                    "application/json",
                    on_click="ignore",
                )
                st.caption("Download buttons export only this inspection to your own device.")
            except (VisualGuardError, OSError) as exc:
                st.error(str(exc))
with evidence_tab:
    if models:
        meta = models[selection][2]
        st.subheader("Trace this model to its data")
        st.json(
            {
                "model_id": meta["model_id"],
                "dataset": meta["provenance"]["dataset"],
                "configuration": meta["config"],
                "fit": meta["fit"],
                "calibration": {
                    key: value
                    for key, value in meta["calibration"].items()
                    if key != "image_scores"
                },
            }
        )
        st.caption(
            "Normal validation score reference. These are not defective validation examples."
        )
        st.bar_chart(
            meta["calibration"]["image_scores"],
            x_label="Validation image index",
            y_label="Anomaly score",
        )
        st.caption(
            "Measured test results and hardware details are in the repository's reports directory."
        )
    else:
        st.write("Model provenance and validation scores appear after an artifact is installed.")
with guide_tab:
    st.markdown("""### A reference for normal appearance
The global baseline compares one frozen ImageNet ResNet-18 embedding per image.
PatchCore compares intermediate image patches with a representative memory bank of normal patches.
Its heatmap highlights unusual regions; it does not establish a defect's physical cause.

### Separate fitting, calibration and evaluation
Normal training images populate the bank. Separate normal images set the threshold.
Official test images are used for evaluation after the model and threshold are frozen.

### Practical limits
Use consistent camera position, category, focus and lighting. A low anomaly score can still miss
a defect. This prototype does not control machinery or certify that a part is safe.
""")
