"""Verify and extract MVTec AD from a local archive or the publisher's download link."""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

DATASET_PAGE = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"
SHA256 = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"
MAX_ARCHIVE = 8 * 1024**3


def download(root: Path, categories: list[str]):
    root.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "mvtec_anomaly_detection.tar.xz"
    partial = archive.with_suffix(".xz.part")
    if not archive.exists():
        # The publisher supplies this link after the user completes its download form.
        url = os.environ.get("MVTEC_AD_DOWNLOAD_URL", "")
        address = urlsplit(url)
        if (
            address.scheme != "https"
            or address.hostname
            not in {"mydrive.ch", "www.mydrive.ch", "mvtec.com", "www.mvtec.com"}
            or address.username
            or address.password
        ):
            raise SystemExit(
                f"Obtain the full MVTec AD archive at {DATASET_PAGE}. Save it as {archive}, "
                "or set MVTEC_AD_DOWNLOAD_URL to the HTTPS download link supplied by the publisher."
            )
        if shutil.disk_usage(root).free < 15 * 1024**3:
            raise RuntimeError("At least 15 GiB free disk required for download and extraction.")
        for attempt in range(4):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                request = urllib.request.Request(  # noqa: S310 -- scheme and publisher host checked above
                    url, headers={"Range": f"bytes={offset}-"} if offset else {}
                )
                with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 -- fixed HTTPS publisher URL
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get("Content-Range", "").startswith(
                        f"bytes {offset}-"
                    ):
                        raise RuntimeError("Unexpected resume response.")
                    size = offset if append else 0
                    last = time.monotonic()
                    with partial.open("ab" if append else "wb") as handle:
                        while chunk := response.read(4 * 1024**2):
                            size += len(chunk)
                            if size > MAX_ARCHIVE:
                                raise RuntimeError("Download exceeded the 8 GiB archive limit.")
                            handle.write(chunk)
                            if time.monotonic() - last > 20:
                                print(f"Downloaded {size / 1024**3:.2f} GiB", flush=True)
                                last = time.monotonic()
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
    candidate = archive if archive.exists() else partial
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError("Official archive SHA-256 mismatch; do not extract.")
    if candidate == partial:
        partial.rename(archive)
    print("Official archive hash verified. Extracting selected categories...", flush=True)
    total, count = 0, 0
    with tarfile.open(archive, "r|xz") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if not name.parts or name.parts[0] not in categories:
                continue
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or ":" in member.name
            ):
                raise RuntimeError("Unsafe archive path.")
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError("Archive links/devices are forbidden.")
            total += member.size
            if total > 10 * 1024**3 or member.size > 20 * 1024**2:
                raise RuntimeError("Extraction size limit exceeded.")
            destination = (root / str(name)).resolve()
            if not destination.is_relative_to(root.resolve()):
                raise RuntimeError("Extraction path escapes dataset root.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError("Archive member has no contents.")
            if destination.exists():
                with source, destination.open("rb") as existing:
                    while chunk := source.read(1024**2):
                        if existing.read(len(chunk)) != chunk:
                            raise RuntimeError(
                                f"Existing file differs from verified archive: {name}"
                            )
                    if existing.read(1):
                        raise RuntimeError(f"Existing file exceeds archive member: {name}")
                continue
            with source, destination.open("xb") as handle:
                shutil.copyfileobj(source, handle)
            count += 1
    (root / "download_provenance.json").write_text(
        json.dumps(
            {
                "source": DATASET_PAGE,
                "sha256": SHA256,
                "categories": categories,
                "extracted_files": count,
                "license": "CC BY-NC-SA 4.0",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Extracted {count} files ({total / 1024**3:.2f} GiB).", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("local_data/MVTecAD"))
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["bottle", "metal_nut", "cable"],
        default=["bottle", "metal_nut", "cable"],
    )
    args = parser.parse_args()
    download(args.root, args.categories)
