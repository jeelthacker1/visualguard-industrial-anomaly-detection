# Security policy

This is a research prototype, not a safety controller. Report security issues
through the repository's private vulnerability reporting feature when available.
Do not include confidential images, credentials, or personal data in a public issue.

## Implemented boundaries

- Uploads are decoded in memory, checked by content, limited to 10 MiB and 16 million
  pixels. Only single-frame opaque RGB/grayscale PNG/JPEG is accepted. Image metadata
  is discarded, and user filenames are never interpreted as filesystem paths.
- No user model uploads, pickle/joblib model loads, shell construction from user input,
  URL fetching from uploads, online training, or test-driven threshold selection.
- Trusted administrator model artifacts use safetensors with SHA-256 integrity checks,
  fixed schemas and a 512 MiB weight-file limit. Hashes detect corruption; they are not
  signatures. Anyone able to replace both an artifact and its integrity manifest is
  already inside the trusted deployment boundary. Keep artifacts read-only.
- API: mandatory secret of at least 32 characters, constant-time comparison, 60 requests
  per minute, one active image, no work queue, five-second body deadline, streaming byte
  limit, timestamp validation, duplicate frame IDs, no-store headers and sanitized failures.
- UI: localhost binding, XSRF/CORS checks, one inference at a time, upload size cap,
  no disk writes of user images. Streamlit's own request/session behavior remains part
  of the dependency trust boundary. The shared UI is not a hardened multi-tenant service.
- Dataset paths must stay under the approved root. The downloader checks the official
  archive digest, restricts members, refuses links/devices, and caps disk extraction.
- Dependency audit and CI are included. Check the date and actual outcome of the
  stored audit; no scan guarantees absence of vulnerabilities.

## Deployment boundary

The API binds to loopback in the documented command. Public GitHub source publication
does not expose the inference service. A public hosted service requires an authenticated
TLS reverse proxy, request-size and connection-rate limits, a non-root process/container,
read-only model volume, secret rotation, monitoring and a separate hosting review.
Keep access logs free of tokens and image data. Do not expose unauthenticated Streamlit
to a public network. Rate limits and idempotency are per process; use one worker or an
external shared limiter. Crashes/reset clear the in-memory 60-second frame-ID cache.

No PLC, actuator, reject gate or robotic command is issued by this project. Failures,
stale frames, missed deadlines, or quality checks route to review; they never imply a
part is certified safe. An unfamiliar product can still receive a low score.
