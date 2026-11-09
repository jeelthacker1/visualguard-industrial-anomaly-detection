import importlib.util
from pathlib import Path

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None, reason="Optional UI not installed"
)
def test_missing_model_is_actionable(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("VISUALGUARD_MODEL_DIR", str(tmp_path / "missing"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")).run(
        timeout=30
    )
    assert not app.exception
    assert any("No trained model" in item.value for item in app.info)
