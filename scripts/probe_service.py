"""Measure the local HTTP path with a trained artifact and a preloaded camera frame."""

import argparse
import json
import secrets
import socket
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import uvicorn

from visualguard.api import create_app
from visualguard.artifacts import load_artifact
from visualguard.metrics import latency_summary
from visualguard.models import configure_torch
from visualguard.utils import write_json


def probe(artifact: Path, frame: Path, output: Path, count: int = 30):
    if output.exists() or not 2 <= count <= 45:
        raise ValueError("Use a new output path and 2–45 timed requests.")
    detector, meta = load_artifact(artifact)
    configure_torch(detector.config)
    data = frame.read_bytes()
    content_type = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
    token = secrets.token_urlsafe(48)
    app = create_app(detector, meta, token=token)
    server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level="error"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        worker = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        worker.start()
        try:
            until = time.monotonic() + 15
            while not server.started:
                if time.monotonic() >= until or not worker.is_alive():
                    raise RuntimeError("Local service did not start.")
                time.sleep(0.05)
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=10
            ) as client:

                def headers(frame_id):
                    return {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": content_type,
                        "X-Frame-Id": frame_id,
                        "X-Captured-At": datetime.now(UTC).isoformat(),
                    }

                elapsed, internal, missed, review = [], [], 0, 0
                for index in range(count + 3):
                    request_headers = headers(f"probe-{index}")
                    start = time.perf_counter()
                    response = client.post("/predict", content=data, headers=request_headers)
                    milliseconds = (time.perf_counter() - start) * 1000
                    response.raise_for_status()
                    result = response.json()
                    if result["model_id"] != meta["model_id"] or result["duplicate_request"]:
                        raise RuntimeError("Service returned an unexpected model or cached frame.")
                    if index >= 3:
                        elapsed.append(milliseconds)
                        internal.append(result["latency_ms"])
                        missed += int(result["deadline_missed"])
                        review += int(result["decision"] == "review_required")
                replay = client.post("/predict", content=data, headers=request_headers)
                replay.raise_for_status()
                if not replay.json()["duplicate_request"]:
                    raise RuntimeError("Replay was processed twice.")
                stale = headers("stale")
                stale["X-Captured-At"] = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
                checks = {
                    "stale_frame": client.post("/predict", content=data, headers=stale).status_code,
                    "invalid_image": client.post(
                        "/predict", content=b"invalid", headers=headers("invalid")
                    ).status_code,
                    "missing_token": client.post("/predict", content=b"x").status_code,
                }
                if checks != {"stale_frame": 422, "invalid_image": 422, "missing_token": 401}:
                    raise RuntimeError(f"Unexpected HTTP rejection behavior {checks}")
                counters = client.get("/metrics", headers=headers("metrics")).json()
        finally:
            server.should_exit = True
            worker.join(timeout=10)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "model_id": meta["model_id"],
        "category": meta["provenance"]["category"],
        "scope": "loopback HTTP from send through response, batch one, same preloaded file with fresh IDs and timestamps",
        "input": frame.name,
        "input_bytes": len(data),
        "warmups": 3,
        "client_round_trip": latency_summary(elapsed),
        "server_inspection": latency_summary(internal),
        "round_trip_milliseconds": elapsed,
        "inspection_milliseconds": internal,
        "inspection_deadline_ms": result["deadline_ms"],
        "inspection_deadline_misses": missed,
        "review_required_responses": review,
        "original_replay_returned": True,
        "rejection_statuses": checks,
        "counters": counters,
        "limits": "Excludes camera capture, plant network, queueing and actuator delay. The service enforces 60 inspection requests per minute.",
    }
    write_json(output, report)
    print(
        json.dumps(
            {"round_trip": report["client_round_trip"], "checks": checks, "deadline_misses": missed}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    probe(args.artifact, args.image, args.output, args.count)
