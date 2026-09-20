"""Download the ONNX model weights.

Weights are deliberately *not* committed (the brief forbids large model files
in the repository). This script is the documented one-command way to get them,
and it verifies a pinned SHA-256 so a reviewer knows they are running exactly
the graph these results were measured on.

    python backend/scripts/fetch_model.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "object_detection_segmentation/ssd-mobilenetv1/model/"
    "ssd_mobilenet_v1_10.onnx"
)
MODEL_PATH = BACKEND_DIR / "models" / "ssd_mobilenet_v1_10.onnx"
EXPECTED_SHA256 = "1fbcf47654165f2e0b5f1bdf3f123b9e9e1128cd6463717767b76ab4b5246f9a"
EXPECTED_BYTES = 29_275_103


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if present"
    )
    args = parser.parse_args()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and not args.force:
        actual = sha256_of(MODEL_PATH)
        if actual == EXPECTED_SHA256:
            print(f"OK  already present and verified: {MODEL_PATH}")
            return 0
        print(f"!!  checksum mismatch on existing file, re-downloading")

    print(f"--  downloading {MODEL_URL}")
    print(f"    -> {MODEL_PATH}  (~{EXPECTED_BYTES / 1e6:.0f} MB)")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - this is a CLI entry point
        print(f"!!  download failed: {exc}", file=sys.stderr)
        print(
            "    The model is SSD-MobileNetV1 from the ONNX Model Zoo. If the "
            "URL has moved, find it under validated/vision/"
            "object_detection_segmentation/ssd-mobilenetv1 at "
            "https://github.com/onnx/models",
            file=sys.stderr,
        )
        return 1

    actual = sha256_of(MODEL_PATH)
    if actual != EXPECTED_SHA256:
        print(
            f"!!  SHA-256 mismatch\n    expected {EXPECTED_SHA256}\n"
            f"    actual   {actual}",
            file=sys.stderr,
        )
        return 1

    print(f"OK  verified SHA-256 {actual}")
    print(f"OK  {MODEL_PATH.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
