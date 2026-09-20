"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { EventList } from "@/components/EventList";
import { StatusPanel } from "@/components/StatusPanel";
import { VideoPanel, type SourceKind } from "@/components/VideoPanel";
import { useEventChannel } from "@/hooks/useEventChannel";
import { api } from "@/lib/api";
import type {
  EventStatus,
  RuleConfig,
  SafetyEvent,
  SystemStatus,
  Zone,
} from "@/lib/types";

const STATUS_POLL_MS = 2000;

export default function Dashboard() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [config, setConfig] = useState<RuleConfig | null>(null);
  const [zone, setZone] = useState<Zone | null>(null);
  const [events, setEvents] = useState<SafetyEvent[]>([]);
  const [filter, setFilter] = useState<EventStatus | "all">("all");
  const [source, setSource] = useState<SourceKind>("webcam");
  const [editingRoi, setEditingRoi] = useState(false);
  const [unreachable, setUnreachable] = useState<string | null>(null);
  const [toast, setToast] = useState<SafetyEvent | null>(null);
  const [highlightId, setHighlightId] = useState<string | null>(null);

  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // --- initial load -------------------------------------------------------

  const loadEvents = useCallback(async () => {
    const list = await api.listEvents(
      filter === "all" ? {} : { status: filter },
    );
    setEvents(list.items);
  }, [filter]);

  useEffect(() => {
    void (async () => {
      try {
        const [cfg, activeZone] = await Promise.all([
          api.config(),
          api.activeZone().catch(() => null),
        ]);
        setConfig(cfg);
        setZone(activeZone);
        setUnreachable(null);
      } catch (err) {
        setUnreachable((err as Error).message);
      }
    })();
  }, []);

  useEffect(() => {
    void loadEvents().catch(() => {
      /* the status poll surfaces connectivity problems */
    });
  }, [loadEvents]);

  // Status is polled rather than pushed: it is a slow-moving health summary,
  // and polling it keeps the WebSocket dedicated to events that matter.
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const next = await api.status();
        if (cancelled) return;
        setStatus(next);
        setUnreachable(null);
        // Adopt server-side zone changes (e.g. another tab redrew the ROI).
        if (next.zone) setZone(next.zone);
      } catch (err) {
        if (!cancelled) setUnreachable((err as Error).message);
      }
    };
    void tick();
    const handle = setInterval(tick, STATUS_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(handle);
    };
  }, []);

  // --- live event channel -------------------------------------------------

  const onChannelMessage = useCallback(
    (event: SafetyEvent, kind: "created" | "updated") => {
      setEvents((current) => {
        const without = current.filter((e) => e.id !== event.id);
        return kind === "created" ? [event, ...without] : [event, ...without].sort(
          (a, b) => b.event_time.localeCompare(a.event_time),
        );
      });

      if (kind === "created") {
        setToast(event);
        setHighlightId(event.id);
        if (toastTimer.current) clearTimeout(toastTimer.current);
        toastTimer.current = setTimeout(() => {
          setToast(null);
          setHighlightId(null);
        }, 6000);
      }
    },
    [],
  );

  const channel = useEventChannel(onChannelMessage);

  // --- actions ------------------------------------------------------------

  const handleRoiDrawn = useCallback(
    async (polygon: [number, number][]) => {
      if (!zone) return;
      try {
        setZone(await api.updateZone(zone.id, polygon));
        setEditingRoi(false);
      } catch (err) {
        setUnreachable((err as Error).message);
      }
    },
    [zone],
  );

  const handleUpdated = useCallback((event: SafetyEvent) => {
    setEvents((current) =>
      current.map((e) => (e.id === event.id ? event : e)),
    );
  }, []);

  const visible =
    filter === "all" ? events : events.filter((e) => e.status === filter);

  return (
    <main className="mx-auto min-h-screen max-w-7xl p-4 lg:p-6">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold text-foreground">
            Local AI Safety Monitoring
          </h1>
          <p className="text-xs text-muted">
            On-device person detection · restricted-zone events · live dashboard
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex overflow-hidden rounded border border-line">
            {(["webcam", "sample"] as SourceKind[]).map((kind) => (
              <button
                key={kind}
                onClick={() => setSource(kind)}
                className={`px-3 py-1.5 text-xs capitalize ${
                  source === kind
                    ? "bg-accent text-accent-foreground"
                    : "text-muted hover:bg-surface-muted"
                }`}
              >
                {kind === "sample" ? "Sample video" : "Webcam"}
              </button>
            ))}
          </div>

          <button
            onClick={() => setEditingRoi((v) => !v)}
            className={`rounded px-3 py-1.5 text-xs ${
              editingRoi
                ? "bg-amber-500 text-slate-950"
                : "border border-line text-muted hover:bg-surface-muted"
            }`}
          >
            {editingRoi ? "Cancel edit" : "Edit zone"}
          </button>
        </div>
      </header>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <VideoPanel
            source={source}
            zone={zone}
            editingRoi={editingRoi}
            onRoiDrawn={handleRoiDrawn}
            onResult={() => {
              /* overlay reads results directly; nothing to lift up here */
            }}
          />
          <StatusPanel
            status={status}
            config={config}
            channel={channel}
            unreachable={unreachable}
          />
        </div>

        <div className="flex max-h-[calc(100vh-8rem)] flex-col">
          <EventList
            events={visible}
            filter={filter}
            onFilterChange={setFilter}
            onUpdated={handleUpdated}
            highlightId={highlightId}
          />
        </div>
      </div>

      {toast && (
        <div className="fixed bottom-5 right-5 z-50 max-w-sm rounded-lg border border-red-600 bg-red-600 p-4 shadow-xl">
          <p className="text-sm font-semibold text-white">
            New safety event
          </p>
          <p className="mt-1 text-xs text-red-50">
            {toast.label} · {(toast.confidence * 100).toFixed(0)}% confidence
          </p>
        </div>
      )}
    </main>
  );
}
