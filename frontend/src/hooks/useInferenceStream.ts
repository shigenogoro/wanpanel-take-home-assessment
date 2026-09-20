"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { WS_BASE } from "@/lib/api";
import type { InferenceMessage, InferenceResult } from "@/lib/types";

/** Frames are downscaled to this width before encoding. Large enough for the
 *  detector, small enough to keep JPEG encode and transfer off the critical
 *  path. */
const CAPTURE_WIDTH = 640;
const JPEG_QUALITY = 0.6;
/** Upper bound on send rate; the backend is usually the real limiter. */
const TARGET_FPS = 12;

export interface StreamStats {
  /** Model time reported by the backend. */
  latencyMs: number;
  /** What the browser actually waited: encode + network + inference + decode. */
  roundTripMs: number;
  fps: number;
  framesSent: number;
}

export type StreamState = "idle" | "connecting" | "streaming" | "error";

/** Either a live <video> or the synthetic sample <canvas>. */
export type CaptureSource = HTMLVideoElement | HTMLCanvasElement;

interface Options {
  videoRef: React.RefObject<CaptureSource | null>;
  source: string;
  enabled: boolean;
  onResult: (result: InferenceResult) => void;
}

/** Intrinsic pixel size of whichever element we are capturing from. */
function naturalSize(element: CaptureSource) {
  return element instanceof HTMLVideoElement
    ? { width: element.videoWidth, height: element.videoHeight, ready: element.readyState >= 2 }
    : { width: element.width, height: element.height, ready: true };
}

/**
 * Pumps frames from a <video> element to /ws/inference.
 *
 * Backpressure is the important part: exactly one frame is in flight at a
 * time. If the CPU slows down, the effective frame rate drops instead of a
 * queue growing without bound and latency climbing forever.
 */
export function useInferenceStream({
  videoRef,
  source,
  enabled,
  onResult,
}: Options) {
  const [state, setState] = useState<StreamState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<StreamStats>({
    latencyMs: 0,
    roundTripMs: 0,
    fps: 0,
    framesSent: 0,
  });

  const socketRef = useRef<WebSocket | null>(null);
  const inFlightRef = useRef(false);
  const sentAtRef = useRef(0);
  const framesSentRef = useRef(0);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;

  const captureFrame = useCallback(async (): Promise<Blob | null> => {
    const element = videoRef.current;
    if (!element) return null;
    const { width, height, ready } = naturalSize(element);
    if (!ready || !width || !height) return null;

    if (!canvasRef.current) canvasRef.current = document.createElement("canvas");
    const canvas = canvasRef.current;
    const scale = CAPTURE_WIDTH / width;
    canvas.width = CAPTURE_WIDTH;
    canvas.height = Math.round(height * scale);

    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(element, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) =>
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", JPEG_QUALITY),
    );
  }, [videoRef]);

  useEffect(() => {
    if (!enabled) {
      socketRef.current?.close();
      socketRef.current = null;
      setState("idle");
      return;
    }

    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    setState("connecting");
    setError(null);
    framesSentRef.current = 0;

    const socket = new WebSocket(
      `${WS_BASE}/ws/inference?source=${encodeURIComponent(source)}`,
    );
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;

    const pump = async () => {
      if (disposed || socket.readyState !== WebSocket.OPEN) return;
      if (inFlightRef.current) return;

      const blob = await captureFrame();
      if (disposed || socket.readyState !== WebSocket.OPEN) return;

      if (!blob) {
        // Video not ready yet -- try again shortly rather than giving up.
        timer = setTimeout(pump, 200);
        return;
      }

      inFlightRef.current = true;
      sentAtRef.current = performance.now();
      socket.send(await blob.arrayBuffer());
      framesSentRef.current += 1;
    };

    socket.onopen = () => {
      setState("streaming");
      void pump();
    };

    socket.onmessage = (raw) => {
      let message: InferenceMessage;
      try {
        message = JSON.parse(raw.data) as InferenceMessage;
      } catch {
        return;
      }

      if (message.type === "error") {
        // Per-frame problems are recoverable; the socket stays open and we
        // keep pumping. Fatal ones arrive with the socket closing right after.
        setError(message.message);
        inFlightRef.current = false;
        timer = setTimeout(pump, 1000 / TARGET_FPS);
        return;
      }

      const roundTripMs = performance.now() - sentAtRef.current;
      inFlightRef.current = false;
      setError(null);
      setStats({
        latencyMs: message.latency_ms,
        roundTripMs: Math.round(roundTripMs),
        fps: message.fps,
        framesSent: framesSentRef.current,
      });
      onResultRef.current(message);

      // Respect the target rate, but never send before the last reply landed.
      const wait = Math.max(0, 1000 / TARGET_FPS - roundTripMs);
      timer = setTimeout(pump, wait);
    };

    socket.onclose = (closeEvent) => {
      if (disposed) return;
      setState("error");
      if (closeEvent.reason) setError(closeEvent.reason);
    };

    socket.onerror = () => {
      if (!disposed) setState("error");
    };

    return () => {
      disposed = true;
      if (timer) clearTimeout(timer);
      inFlightRef.current = false;
      socket.close();
      socketRef.current = null;
    };
  }, [enabled, source, captureFrame]);

  return { state, error, stats };
}
