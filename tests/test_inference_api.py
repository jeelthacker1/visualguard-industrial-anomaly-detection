from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from visualguard.api import create_app
from visualguard.errors import VisualGuardError
from visualguard.inference import inspect_bytes, validate_capture_timestamp

TOKEN = "test-only-" + "x" * 32


def headers(frame="frame-1"):
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "image/png",
        "X-Frame-Id": frame,
        "X-Captured-At": datetime.now(UTC).isoformat(),
    }


def test_inference_contract_and_deadline(fake_model, png):
    detector, meta = fake_model
    result, image, anomaly_map = inspect_bytes(png, detector, meta)
    assert result["decision"] == "normal_appearance"
    assert result["latency_ms"] > 0 and not result["score_is_probability"]
    assert image.size == (64, 64) and anomaly_map.shape == (32, 32)
    result, _, _ = inspect_bytes(png, detector, meta, deadline_ms=0.000001)
    assert result["decision"] == "review_required" and result["deadline_missed"]


def test_quality_drift_forces_review(fake_model, png):
    detector, meta = fake_model
    meta["provenance"]["quality_envelope"] = {"brightness": [0.9, 1.0]}
    result, _, _ = inspect_bytes(png, detector, meta)
    assert result["decision"] == "review_required"
    assert result["anomaly_above_threshold"] is False


@pytest.mark.parametrize(
    "value", ["bad", "2026-01-01T00:00:00", "2000-01-01T00:00:00Z", "2099-01-01T00:00:00Z"]
)
def test_stale_and_bad_timestamps(value):
    with pytest.raises(VisualGuardError):
        validate_capture_timestamp(value)


def test_api_auth_duplicate_conflict_and_bad_upload(fake_model, png):
    client = TestClient(create_app(*fake_model, token=TOKEN))
    assert client.get("/health").json()["status"] == "ready"
    assert client.post("/predict", content=png).status_code == 401
    assert client.post("/predict", content=png, headers=headers()).status_code == 200
    duplicate = client.post("/predict", content=png, headers=headers())
    assert duplicate.json()["duplicate_request"]
    assert client.post("/predict", content=png + b"modified", headers=headers()).status_code == 409
    assert client.post("/predict", content=b"bad", headers=headers("frame-2")).status_code == 422
    assert (
        client.post(
            "/predict", content=png, headers={**headers(), "Content-Type": "text/plain"}
        ).status_code
        == 415
    )
    stale = (datetime.now(UTC) - timedelta(seconds=30)).isoformat()
    assert (
        client.post(
            "/predict", content=png, headers={**headers(), "X-Captured-At": stale}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predict", content=png, headers={**headers(), "Content-Length": str(11 * 1024**2)}
        ).status_code
        == 413
    )
    assert client.get("/metrics", headers=headers()).json()["processed"] == 1


def test_api_fails_closed_on_model_error(fake_model, png):
    detector, meta = fake_model

    def broken(_):
        raise RuntimeError("internal secret path should not reach a client")

    detector.predict = broken
    client = TestClient(create_app(detector, meta, token=TOKEN))
    response = client.post("/predict", content=png, headers=headers())
    assert response.status_code == 503
    assert "secret" not in response.text


def test_api_requires_secret(fake_model):
    with pytest.raises(VisualGuardError):
        create_app(*fake_model, token="short")


def test_duplicate_cannot_refresh_old_capture_timestamp(fake_model, png, monkeypatch):
    import visualguard.api as api

    now = datetime.now(UTC)
    current = [now]
    monkeypatch.setattr(
        api,
        "validate_capture_timestamp",
        lambda value: validate_capture_timestamp(value, now=current[0]),
    )
    with TestClient(create_app(*fake_model, token=TOKEN)) as client:
        original_headers = {**headers(), "X-Captured-At": now.isoformat()}
        assert client.post("/predict", content=png, headers=original_headers).status_code == 200
        current[0] = now + timedelta(seconds=3)
        refreshed_headers = {**headers(), "X-Captured-At": current[0].isoformat()}
        assert client.post("/predict", content=png, headers=refreshed_headers).status_code == 422
