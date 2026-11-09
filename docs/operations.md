# Running and operating VisualGuard

## Offline research

Use Python 3.13 and the tested lockfile. The blueprint proposed Python 3.11; the local host
has 3.13, so that is the declared and tested target. The downloader requires about 15 GiB
free before extracting the three categories. The final subset is much smaller than the full
archive. Keep `local_data`, `artifacts`, `.cache` and `runs` private and outside Git.

`visualguard prepare` creates the audit and split once. If it fails, inspect the audit before
changing files. A new manifest path is required after legitimate source changes. Use the CLI
to fit, evaluate and generate reports. The UI never trains or edits evaluation inputs.

The study script resumes by checking completed artifacts and result files. If an evaluation
started and crashed, inspect the failure, then run an explicit `--allow-repeat` diagnostic
with a new output directory. Do not delete provenance markers to disguise repeated testing.

## Dashboard

Run `python -m streamlit run app/streamlit_app.py`. It defaults to loopback port 8501.
`VISUALGUARD_MODEL_DIR` selects an administrator-controlled artifact directory; the user
cannot supply an arbitrary model path. Model resources are cached and inference is locked
across sessions. Uploads stay in memory; JSON/overlay downloads are explicit user actions.
Choose the correct category, consistent lighting, a focused image and a comparable crop.

## API / frame ingestion

Set `VISUALGUARD_ARTIFACT` and a random `VISUALGUARD_API_TOKEN` of at least 32 characters
locally, then run:

```text
python -m uvicorn visualguard.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

POST raw PNG/JPEG bytes to `/predict` with `Content-Type: image/png` (or image/jpeg),
`Authorization: Bearer <local secret>`, `X-Frame-Id: <unique capture identifier>` and
`X-Captured-At: <timezone-aware ISO-8601 UTC timestamp>`. `/health` exposes readiness,
model ID and busy status; `/metrics` requires the token and returns anonymous counters.
The `rejected` counter covers errors after acquisition of the inspection slot. Authentication,
timestamp and rate checks performed before acquisition are visible in their HTTP status codes.
The API returns scalar inspection metadata; localization is available in the dashboard/CLI
model interface, not a large unaudited JSON map response.

Frames older than two seconds or over 500 ms in the future are rejected. Synchronize clocks.
The timestamp is checked again after decoding and inference. A deadline overrun produces
review_required; Python CPU work is not forcibly interrupted mid-inference. An upstream
supervisor must implement a physical deadline if the application needs one.

Frame IDs remain in the replay cache for up to 60 seconds. Identical bytes return the existing
result only while the original capture is still within the two-second freshness limit,
with `duplicate_request=true` and the original timestamps. A new timestamp cannot refresh
an old cached capture. Different bytes with the same ID
return 409. Frame identity is a caller responsibility; this cache is not durable exactly-once
delivery. New frame IDs with repeated content are new captures. Raw image bytes are never
kept in the replay cache. Only hashes and scalar results are retained, bounded to 512 entries.

One request can read/decode/infer at a time. Busy responses use 429, with no unbounded queue.
Clients should discard obsolete frames rather than repeatedly retry stale ones. Upload bodies
are limited to five seconds. Error responses expose no internal exception or filesystem path.

## Deployment and rollback

A public repository contains source and evidence, not a public inference server. GitHub Pages
cannot execute PyTorch/Streamlit. The included Dockerfile is an optional local packaging route.
It runs as a non-root user and expects a read-only artifact volume. Docker has not been assumed
installed. Do not describe an image as tested unless its build/runtime check has actually run.

For a hosted service, provision TLS/authentication, connection limits, audit retention and
operator access separately. Use one worker for in-memory rate/idempotency semantics. Version
artifacts by directory and SHA; replace deployment configuration to roll back, never mutate a
running model. Keep raw production images only under a separately approved retention policy.

## Failure response

| Condition | Behavior | Operator action |
|---|---|---|
| Invalid upload or oversized body | Rejected | Supply a valid bounded image |
| Missing/tampered artifact | Startup/load fails | Restore a verified artifact |
| Wrong category | User must select correct model | Confirm acquisition/product routing |
| Stale clock/frame | Rejected or review | Synchronize clocks and discard frame |
| Busy service/rate limit | 429 | Apply upstream backpressure |
| Internal model error | 503 | Manual review and investigate privately |
| Deadline or quality drift | Review required | Check camera/process; do not auto-accept |
| Shifted production distribution | May still be missed | Collect independent data and revalidate |
