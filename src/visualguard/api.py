"""Bounded single-worker inspection API; no upload storage or online training."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
from collections import OrderedDict, deque
from pathlib import Path
from time import monotonic

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .errors import VisualGuardError
from .images import MAX_UPLOAD_BYTES
from .inference import inspect_bytes, validate_capture_timestamp


def create_app(detector=None, metadata=None, *, token: str | None = None) -> FastAPI:
    token = token or os.environ.get("VISUALGUARD_API_TOKEN", "")
    if len(token) < 32:
        raise VisualGuardError(
            "Set VISUALGUARD_API_TOKEN to a random secret of at least 32 characters."
        )
    if detector is None:
        from .artifacts import load_artifact
        from .models import configure_torch

        artifact = os.environ.get("VISUALGUARD_ARTIFACT")
        if not artifact:
            raise VisualGuardError(
                "Set VISUALGUARD_ARTIFACT to an administrator-provisioned model directory."
            )
        detector, metadata = load_artifact(Path(artifact))
        configure_torch(detector.config)
    app = FastAPI(
        title="VisualGuard inspection API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    semaphore = asyncio.Semaphore(1)
    seen: OrderedDict = OrderedDict()
    arrivals = deque()
    counts = {"processed": 0, "rejected": 0, "duplicates": 0, "deadline_misses": 0}

    def authorize(request):
        candidate = request.headers.get("Authorization", "")
        if not hmac.compare_digest(candidate.encode(), f"Bearer {token}".encode()):
            raise HTTPException(401, "Valid bearer token required.")

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
        return response

    @app.get("/health")
    async def health():
        return {"status": "ready", "model_id": metadata["model_id"], "busy": semaphore.locked()}

    @app.get("/metrics")
    async def counters(request: Request):
        authorize(request)
        return dict(counts)

    @app.post("/predict")
    async def predict(request: Request):
        authorize(request)
        now = monotonic()
        while arrivals and now - arrivals[0] >= 60:
            arrivals.popleft()
        if len(arrivals) >= 60:
            raise HTTPException(429, "Rate limit: 60 inspection requests per minute per service.")
        arrivals.append(now)
        if request.headers.get("content-type", "").split(";")[0] not in {"image/png", "image/jpeg"}:
            raise HTTPException(415, "Send raw PNG/JPEG bytes with the correct Content-Type.")
        try:
            declared = int(request.headers.get("content-length", "0"))
            if declared < 0 or declared > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Image exceeds upload limit.")
        except ValueError as exc:
            raise HTTPException(400, "Malformed Content-Length.") from exc
        frame_id = request.headers.get("x-frame-id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", frame_id):
            raise HTTPException(
                400, "X-Frame-Id is required (1–64 letters, digits, dash or underscore)."
            )
        captured_at = request.headers.get("x-captured-at")
        if captured_at is None:
            raise HTTPException(
                400, "X-Captured-At with a timezone is required; synchronize the camera clock."
            )
        try:
            validate_capture_timestamp(captured_at)
        except VisualGuardError as exc:
            raise HTTPException(422, str(exc)) from exc
        # Acquire before buffering; concurrent large bodies cannot queue in process memory.
        if semaphore.locked():
            raise HTTPException(429, "Inspector busy; discard this frame or retry a fresh frame.")
        await semaphore.acquire()
        try:
            data = bytearray()
            try:
                async with asyncio.timeout(5):
                    async for chunk in request.stream():
                        if len(data) + len(chunk) > MAX_UPLOAD_BYTES:
                            raise HTTPException(413, "Image exceeds upload limit.")
                        data.extend(chunk)
            except TimeoutError as exc:
                raise HTTPException(408, "Upload exceeded five seconds.") from exc
            validate_capture_timestamp(captured_at)
            digest = hashlib.sha256(data).hexdigest()
            while seen and now - next(iter(seen.values()))[0] > 60:
                seen.popitem(last=False)
            if frame_id in seen:
                _, old_digest, result = seen[frame_id]
                if old_digest != digest:
                    raise HTTPException(409, "Frame ID reused with different image contents.")
                validate_capture_timestamp(result.get("captured_at"))
                counts["duplicates"] += 1
                return {**result, "duplicate_request": True}
            task = asyncio.create_task(
                asyncio.to_thread(
                    inspect_bytes, bytes(data), detector, metadata, captured_at=captured_at
                )
            )
            try:
                result, _, _ = await asyncio.shield(task)
            except asyncio.CancelledError:
                # A disconnected client must not release capacity while CPU work continues.
                await task
                raise
            counts["processed"] += 1
            counts["deadline_misses"] += int(result["deadline_missed"])
            result["frame_id"] = frame_id
            result["duplicate_request"] = False
            seen[frame_id] = (monotonic(), digest, result)
            while len(seen) > 512:
                seen.popitem(last=False)
            return result
        except VisualGuardError as exc:
            counts["rejected"] += 1
            raise HTTPException(422, str(exc)) from exc
        except HTTPException:
            counts["rejected"] += 1
            raise
        except Exception:
            counts["rejected"] += 1
            return JSONResponse(
                status_code=503,
                content={"detail": "Inspection unavailable; route the part for manual review."},
            )
        finally:
            semaphore.release()

    return app
