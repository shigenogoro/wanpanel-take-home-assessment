"""Generate the two evaluation clips.

The brief requires evaluating one scenario that should trigger and one that
should not, using "public, synthetic, or self-recorded" material. These clips
are synthesised from a single public detection sample image: a 640x480 window
onto the lower half of the photo is shifted horizontally, so real photographed
people move through the frame at real scale against real background texture.

The window is taken from CROP_Y downwards rather than from the top, because the
rule tests a person's foot point. A crop that cuts people off at the waist puts
every foot point on the frame's bottom edge (y ~= 1.0), where it is both
invisible in the overlay and impossible to separate from "outside the zone".

Why synthesise rather than ship recorded video:

* the clips are reproducible -- a reviewer regenerates byte-identical input and
  gets the same numbers, instead of trusting a video they cannot verify;
* no video files need committing to the repository;
* no real people are recorded, which keeps the "no real patient data" rule
  trivially satisfied.

They are a regression fixture, not a substitute for a live demo. Record real
webcam footage for the demo video.

    python backend/scripts/make_fixtures.py
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
FIXTURES = BACKEND_DIR / "fixtures"
#: The dashboard's "Sample" source animates this same image in the browser.
FRONTEND_PUBLIC = BACKEND_DIR.parent / "frontend" / "public"

#: A widely-used public object-detection sample image. Chosen over the other
#: common sample (zidane.jpg) because it shows people full-length, standing on
#: a pavement: their feet are in frame, which is what the zone rule needs.
SOURCE_URL = "https://ultralytics.com/images/bus.jpg"
SOURCE_IMAGE = FIXTURES / "source_people.jpg"

FPS = 10
WIDTH, HEIGHT = 640, 480

#: Top edge of the 640x480 window into the source photo. At this offset the
#: detector sees two full-length people with foot points around y = 0.78-0.85,
#: clear of both the frame edge and the ROI boundary.
CROP_Y = 500

#: The ROI both clips are evaluated against. Matches EVAL_ZONE in evaluate.py.
#: At rest both foot points sit near x = 0.21 and x = 0.44, inside this box;
#: held to the right (shift ~250) they sit near x = 0.60 and x = 0.85, outside
#: it with margin to spare.
EVAL_ZONE = [[0.08, 0.55], [0.48, 0.55], [0.48, 1.0], [0.08, 1.0]]


def ensure_source() -> np.ndarray:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    if not SOURCE_IMAGE.exists():
        print(f"--  downloading source image {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, SOURCE_IMAGE)
    image = cv2.imread(str(SOURCE_IMAGE))
    if image is None:
        raise SystemExit(f"could not read {SOURCE_IMAGE}")

    # The browser's Sample source animates the same photo, so give the frontend
    # its own copy rather than committing a third-party image to the repo.
    if FRONTEND_PUBLIC.is_dir():
        shutil.copyfile(SOURCE_IMAGE, FRONTEND_PUBLIC / "source_people.jpg")
        print(f"OK  copied source image to {FRONTEND_PUBLIC.name}/")
    return image


def make_frame(source: np.ndarray, pan_x: int, shift: int) -> np.ndarray:
    """One frame: a window onto the photo, shifted sideways.

    `pan_x` selects the window; `shift` slides its content horizontally, which
    is what moves the person's foot point across the normalised x axis. The gap
    left behind is filled by replicating the edge so no hard black bar appears
    that the detector might latch onto.
    """
    height, width = source.shape[:2]
    pan_x = int(np.clip(pan_x, 0, max(0, width - WIDTH)))
    top = int(np.clip(CROP_Y, 0, max(0, height - HEIGHT)))
    window = source[top : top + HEIGHT, pan_x : pan_x + WIDTH].copy()

    if shift == 0:
        return window

    canvas = np.empty_like(window)
    if shift > 0:
        canvas[:, shift:] = window[:, : WIDTH - shift]
        canvas[:, :shift] = window[:, :1]
    else:
        gap = -shift
        canvas[:, : WIDTH - gap] = window[:, gap:]
        canvas[:, WIDTH - gap :] = window[:, -1:]
    return canvas


def write_clip(path: Path, frames: list[np.ndarray]) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise SystemExit(f"could not open a video writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"OK  {path.name}  {len(frames)} frames  {len(frames) / FPS:.1f}s")


def build_positive(source: np.ndarray) -> list[np.ndarray]:
    """Person approaches from the right, enters the ROI, and stays 4 s.

    4 s of dwell against a 2 s threshold leaves comfortable margin, so the
    clip does not become flaky if a frame or two is missed.
    """
    frames: list[np.ndarray] = []
    # 1.5 s clearly outside the zone (content pushed to the right).
    for _ in range(15):
        frames.append(make_frame(source, 0, 250))
    # 1.5 s walking in.
    for step in range(15):
        shift = int(250 - (250 * (step + 1) / 15))
        frames.append(make_frame(source, 0, shift))
    # 4 s standing inside. The few pixels of pan keep the boxes moving
    # slightly, so the tracker is doing real matching rather than seeing a
    # frozen image.
    for step in range(40):
        frames.append(make_frame(source, step % 6, 0))
    return frames


def build_negative(source: np.ndarray) -> list[np.ndarray]:
    """Person crosses the frame but never sets foot inside the ROI.

    Held well to the right of the zone for the whole clip -- longer than the
    dwell threshold, so the clip proves the *zone* test is doing the work and
    not merely that the clip was too short to trigger.
    """
    frames: list[np.ndarray] = []
    for step in range(70):  # 7 s, comfortably past the 2 s dwell window
        shift = 250 + int(25 * np.sin(step / 6.0))
        frames.append(make_frame(source, step % 12, shift))
    return frames


def main() -> int:
    source = ensure_source()
    print(f"--  source {SOURCE_IMAGE.name} {source.shape[1]}x{source.shape[0]}")
    write_clip(FIXTURES / "positive_zone_entry.mp4", build_positive(source))
    write_clip(FIXTURES / "negative_walk_by.mp4", build_negative(source))
    print("\nNext: python backend/scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
