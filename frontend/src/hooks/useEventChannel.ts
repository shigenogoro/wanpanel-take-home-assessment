"use client";

import { useEffect, useRef, useState } from "react";

import { WS_BASE } from "@/lib/api";
import type { EventChannelMessage, SafetyEvent } from "@/lib/types";

export type ConnectionState = "connecting" | "open" | "closed";

/**
 * Subscribes to /ws/events so the dashboard updates without a page refresh.
 *
 * Reconnects with backoff, because a backend restart during a demo should heal
 * itself rather than leave a silently dead dashboard.
 */
export function useEventChannel(onMessage: (event: SafetyEvent, kind: "created" | "updated") => void) {
  const [state, setState] = useState<ConnectionState>("connecting");
  const handlerRef = useRef(onMessage);
  handlerRef.current = onMessage;

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      setState("connecting");
      socket = new WebSocket(`${WS_BASE}/ws/events`);

      socket.onopen = () => {
        attempt = 0;
        setState("open");
      };

      socket.onmessage = (raw) => {
        try {
          const message = JSON.parse(raw.data) as EventChannelMessage;
          if (message.type === "event.created") {
            handlerRef.current(message.payload, "created");
          } else if (message.type === "event.updated") {
            handlerRef.current(message.payload, "updated");
          }
        } catch {
          /* ignore anything that is not a message we understand */
        }
      };

      socket.onclose = () => {
        setState("closed");
        if (disposed) return;
        attempt += 1;
        const delay = Math.min(1000 * 2 ** (attempt - 1), 10_000);
        retryTimer = setTimeout(connect, delay);
      };

      socket.onerror = () => socket?.close();
    };

    connect();

    return () => {
      disposed = true;
      if (retryTimer) clearTimeout(retryTimer);
      socket?.close();
    };
  }, []);

  return state;
}
