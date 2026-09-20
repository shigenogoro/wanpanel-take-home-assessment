"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  useInferenceStream,
  type CaptureSource,
} from "@/hooks/useInferenceStream";
import type { Detection, InferenceResult, Zone } from "@/lib/types";

export type SourceKind = "webcam" | "sample";

const SAMPLE_IMAGE = "/source_people.jpg";
const SAMPLE_WIDTH = 640;
const SAMPLE_HEIGHT = 480;
/**
 * Top edge of the window into the source photo. Must match CROP_Y in
 * backend/scripts/make_fixtures.py: cropping from the top would cut the people
 * off at the waist, putting every foot point on the frame's bottom edge where
 * the zone rule can neither see it nor draw it.
 */
const SAMPLE_CROP_Y = 500;

interface Props {
  source: SourceKind;
  zone: Zone | null;
  editingRoi: boolean;
  onRoiDrawn: (polygon: [number, number][]) => void;
  onResult: (result: InferenceResult) => void;
}

/** Human-readable reasons for the two camera failures the brief calls out. */
function describeCameraError(error: unknown): string {
  const name = (error as DOMException)?.name;
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Camera permission was denied. Allow camera access in your browser and press Retry — or switch the source to Sample.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No camera device was found. Connect a webcam and press Retry — or switch the source to Sample.";
  }
  if (name === "NotReadableError") {
    return "The camera is in use by another application. Close it and press Retry.";
  }
  return `Could not start the camera: ${(error as Error)?.message ?? String(error)}`;
}

export function VideoPanel({
  source,
  zone,
  editingRoi,
  onRoiDrawn,
  onResult,
}: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const sampleRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const captureRef = useRef<CaptureSource | null>(null);
  const latestRef = useRef<InferenceResult | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [cameraError, setCameraError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [retryToken, setRetryToken] = useState(0);
  const [draft, setDraft] = useState<{
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  } | null>(null);

  const handleResult = useCallback(
    (result: InferenceResult) => {
      latestRef.current = result;
      onResult(result);
    },
    [onResult],
  );

  const { state, error, stats } = useInferenceStream({
    videoRef: captureRef,
    source: source === "webcam" ? "webcam" : "video:synthetic_sample",
    enabled: ready,
    onResult: handleResult,
  });

  // --- webcam -------------------------------------------------------------

  useEffect(() => {
    if (source !== "webcam") return;

    let cancelled = false;
    setReady(false);
    setCameraError(null);

    const video = videoRef.current;
    if (!video) return;

    navigator.mediaDevices
      ?.getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        video.srcObject = stream;
        video.muted = true;
        return video.play().then(() => {
          captureRef.current = video;
          setReady(true);
        });
      })
      .catch((err) => !cancelled && setCameraError(describeCameraError(err)));

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [source, retryToken]);

  // --- synthetic sample ---------------------------------------------------

  /**
   * The sample source is animated in the browser rather than played from a
   * video file. OpenCV can only write MPEG-4 Part 2 here (no H.264 encoder),
   * which Chrome refuses to decode — so instead of shipping a clip the browser
   * cannot play, a 640x480 window pans across a still photo on a canvas. Same
   * motion, same real person, and no codec dependency for the reviewer.
   */
  useEffect(() => {
    if (source !== "sample") return;

    let cancelled = false;
    let frame = 0;
    let step = 0;
    setReady(false);
    setCameraError(null);

    const image = new Image();
    image.src = SAMPLE_IMAGE;

    image.onerror = () =>
      !cancelled && setCameraError(`Could not load ${SAMPLE_IMAGE}`);

    image.onload = () => {
      if (cancelled) return;
      const canvas = sampleRef.current;
      if (!canvas) return;
      canvas.width = SAMPLE_WIDTH;
      canvas.height = SAMPLE_HEIGHT;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      captureRef.current = canvas;
      setReady(true);

      const cropY = Math.min(SAMPLE_CROP_Y, Math.max(0, image.height - SAMPLE_HEIGHT));

      const draw = () => {
        if (cancelled) return;
        frame = requestAnimationFrame(draw);
        step += 1;

        // 12 s loop: approach from the right, settle inside the zone, repeat.
        // 250px of shift matches the fixture clips' "outside" hold.
        const t = (step % 720) / 720;
        // No horizontal pan: this photo has only ~170px of slack and panning
        // drags the foot points across the ROI edge. In particular do not derive
        // it from `step`, which advances every rAF tick and shakes the image.
        const panX = 0;
        // Shift the content sideways to walk the person across the frame.
        const shift = Math.round(250 * Math.max(0, 1 - t * 3));

        ctx.drawImage(
          image,
          panX,
          cropY,
          SAMPLE_WIDTH,
          SAMPLE_HEIGHT,
          shift,
          0,
          SAMPLE_WIDTH,
          SAMPLE_HEIGHT,
        );
        if (shift > 0) {
          // Replicate the left edge so no hard black bar appears.
          ctx.drawImage(
            image,
            panX,
            cropY,
            2,
            SAMPLE_HEIGHT,
            0,
            0,
            shift,
            SAMPLE_HEIGHT,
          );
        }
      };
      frame = requestAnimationFrame(draw);
    };

    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
    };
  }, [source, retryToken]);

  // --- overlay rendering --------------------------------------------------

  useEffect(() => {
    let frame = 0;

    const draw = () => {
      frame = requestAnimationFrame(draw);
      const canvas = overlayRef.current;
      const host = source === "webcam" ? videoRef.current : sampleRef.current;
      if (!canvas || !host) return;

      const width = host.clientWidth;
      const height = host.clientHeight;
      if (!width || !height) return;
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      const result = latestRef.current;
      const occupied = (result?.zone_state.occupied_track_ids.length ?? 0) > 0;

      drawZone(ctx, zone, draft, width, height, occupied);
      result?.detections.forEach((d) => drawDetection(ctx, d, width, height));
    };

    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [zone, draft, source]);

  // --- ROI drawing --------------------------------------------------------

  const pointerPosition = (event: React.PointerEvent) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
    };
  };

  const onPointerDown = (event: React.PointerEvent) => {
    if (!editingRoi) return;
    const { x, y } = pointerPosition(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    setDraft({ x1: x, y1: y, x2: x, y2: y });
  };

  const onPointerMove = (event: React.PointerEvent) => {
    if (!editingRoi || !draft) return;
    const { x, y } = pointerPosition(event);
    setDraft({ ...draft, x2: x, y2: y });
  };

  const onPointerUp = () => {
    if (!editingRoi || !draft) return;
    const x1 = Math.min(draft.x1, draft.x2);
    const x2 = Math.max(draft.x1, draft.x2);
    const y1 = Math.min(draft.y1, draft.y2);
    const y2 = Math.max(draft.y1, draft.y2);
    setDraft(null);
    // Ignore an accidental click that produced a near-zero rectangle.
    if (x2 - x1 < 0.02 || y2 - y1 < 0.02) return;
    onRoiDrawn([
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2],
    ]);
  };

  const blocking = cameraError ?? (state === "error" ? error : null);

  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-lg border border-line bg-black">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`block h-auto w-full ${source === "webcam" ? "" : "hidden"}`}
        />
        <canvas
          ref={sampleRef}
          className={`block h-auto w-full ${source === "sample" ? "" : "hidden"}`}
        />
        <canvas
          ref={overlayRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          className={`absolute inset-0 h-full w-full ${
            editingRoi ? "cursor-crosshair" : "pointer-events-none"
          }`}
        />

        <div className="absolute left-3 top-3 rounded bg-black/70 px-3 py-2 font-mono text-xs text-slate-100">
          <div>
            {state === "streaming" ? "● live" : `○ ${state}`} ·{" "}
            {source === "webcam" ? "webcam" : "sample"}
          </div>
          <div>
            {stats.fps.toFixed(1)} fps · model {stats.latencyMs.toFixed(0)} ms ·
            round-trip {stats.roundTripMs} ms
          </div>
          <div>frames sent {stats.framesSent}</div>
        </div>

        {editingRoi && (
          <div className="absolute right-3 top-3 rounded bg-amber-500/90 px-3 py-1 text-xs font-semibold text-black">
            Drag to redraw the restricted zone
          </div>
        )}

        {blocking && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/85 p-6">
            <div className="max-w-md space-y-3 text-center">
              <p className="text-sm text-amber-300">{blocking}</p>
              <button
                onClick={() => setRetryToken((n) => n + 1)}
                className="rounded bg-slate-200 px-4 py-1.5 text-sm font-medium text-slate-900 hover:bg-white"
              >
                Retry
              </button>
            </div>
          </div>
        )}
      </div>

      {error && state !== "error" && (
        <p className="text-xs text-warn">{error}</p>
      )}
    </div>
  );
}

// --- canvas helpers -------------------------------------------------------

function drawZone(
  ctx: CanvasRenderingContext2D,
  zone: Zone | null,
  draft: { x1: number; y1: number; x2: number; y2: number } | null,
  width: number,
  height: number,
  occupied: boolean,
) {
  const polygon: [number, number][] = draft
    ? [
        [draft.x1, draft.y1],
        [draft.x2, draft.y1],
        [draft.x2, draft.y2],
        [draft.x1, draft.y2],
      ]
    : (zone?.polygon ?? []);

  if (polygon.length < 3) return;

  ctx.beginPath();
  polygon.forEach(([x, y], index) => {
    const px = x * width;
    const py = y * height;
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();

  // Red while someone is inside, so the trigger condition is visible at a
  // glance in the demo video.
  // Yellow while clear, red while occupied. Yellow rather than blue so the ROI
  // never reads as "another detection box" beside the green ones.
  ctx.fillStyle = occupied ? "rgba(239,68,68,0.28)" : "rgba(234,179,8,0.16)";
  ctx.strokeStyle = occupied ? "rgb(239,68,68)" : "rgb(234,179,8)";
  ctx.lineWidth = 2;
  ctx.fill();
  ctx.stroke();
}

function drawDetection(
  ctx: CanvasRenderingContext2D,
  detection: Detection,
  width: number,
  height: number,
) {
  const [x1, y1, x2, y2] = detection.bbox;
  const left = x1 * width;
  const top = y1 * height;
  const boxWidth = (x2 - x1) * width;
  const boxHeight = (y2 - y1) * height;

  const colour = detection.in_zone ? "rgb(239,68,68)" : "rgb(34,197,94)";
  ctx.strokeStyle = colour;
  ctx.lineWidth = 2;
  ctx.strokeRect(left, top, boxWidth, boxHeight);

  const label = `#${detection.track_id} ${(detection.confidence * 100).toFixed(0)}%`;
  ctx.font = "12px ui-monospace, monospace";
  const textWidth = ctx.measureText(label).width;
  ctx.fillStyle = colour;
  ctx.fillRect(left, Math.max(0, top - 18), textWidth + 10, 18);
  ctx.fillStyle = "#0f172a";
  ctx.fillText(label, left + 5, Math.max(12, top - 5));

  // The foot point is what the zone rule actually tests, so show it.
  const [footX, footY] = detection.foot;
  ctx.beginPath();
  ctx.arc(footX * width, footY * height, 4, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();
}
