"use client";

import { useState } from "react";

import { api } from "@/lib/api";
import type { EventStatus, SafetyEvent } from "@/lib/types";

interface Props {
  events: SafetyEvent[];
  filter: EventStatus | "all";
  onFilterChange: (filter: EventStatus | "all") => void;
  onUpdated: (event: SafetyEvent) => void;
  highlightId: string | null;
}

const STATUS_STYLES: Record<EventStatus, string> = {
  open: "border-red-500 bg-red-500/10 text-danger",
  acknowledged: "border-amber-500 bg-amber-500/10 text-warn",
  resolved: "border-emerald-500 bg-emerald-500/10 text-ok",
};

const TYPE_LABELS: Record<string, string> = {
  restricted_zone_entry: "Restricted-zone entry",
  person_absence: "Person absence",
  manual_test: "Manual test (debug)",
};

function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function EventCard({
  event,
  onUpdated,
  highlighted,
}: {
  event: SafetyEvent;
  onUpdated: (event: SafetyEvent) => void;
  highlighted: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSnapshot, setShowSnapshot] = useState(false);

  const transition = async (status: EventStatus) => {
    setBusy(true);
    setError(null);
    try {
      onUpdated(await api.setEventStatus(event.id, status));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const dwell = event.reason?.["dwell_seconds"];

  return (
    <li
      className={`rounded-lg border p-3 transition-colors ${
        highlighted
          ? "border-red-500 bg-red-500/10"
          : "border-line bg-surface"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-foreground">
            {TYPE_LABELS[event.type] ?? event.type}
          </p>
          <p className="mt-0.5 text-xs text-muted">
            {formatTime(event.event_time)} · {(event.confidence * 100).toFixed(0)}%
            confidence
            {event.track_id !== null && ` · track #${event.track_id}`}
          </p>
        </div>
        <span
          className={`shrink-0 rounded border px-2 py-0.5 text-[11px] font-medium ${
            STATUS_STYLES[event.status]
          }`}
        >
          {event.status}
        </span>
      </div>

      <p className="mt-2 text-[11px] text-subtle">
        {event.source} · {event.model_name} {event.model_version}
        {typeof dwell === "number" && ` · dwelled ${dwell.toFixed(1)}s`}
      </p>

      {showSnapshot && event.has_snapshot && (
        /* eslint-disable-next-line @next/next/no-img-element */
        <img
          src={api.snapshotUrl(event.id)}
          alt={`Snapshot for event ${event.id}`}
          className="mt-2 w-full rounded border border-line"
        />
      )}

      {error && <p className="mt-2 text-xs text-danger">{error}</p>}

      <div className="mt-3 flex flex-wrap gap-2">
        {event.status === "open" && (
          <button
            disabled={busy}
            onClick={() => transition("acknowledged")}
            className="rounded bg-amber-600 px-3 py-1 text-xs font-medium text-white hover:bg-amber-500 disabled:opacity-50"
          >
            Acknowledge
          </button>
        )}
        {event.status !== "resolved" && (
          <button
            disabled={busy}
            onClick={() => transition("resolved")}
            className="rounded bg-emerald-700 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-600 disabled:opacity-50"
          >
            Resolve
          </button>
        )}
        {event.has_snapshot && (
          <button
            onClick={() => setShowSnapshot((v) => !v)}
            className="rounded border border-line px-3 py-1 text-xs text-muted hover:bg-surface-muted"
          >
            {showSnapshot ? "Hide" : "Snapshot"}
          </button>
        )}
      </div>
    </li>
  );
}

export function EventList({
  events,
  filter,
  onFilterChange,
  onUpdated,
  highlightId,
}: Props) {
  const filters: (EventStatus | "all")[] = [
    "all",
    "open",
    "acknowledged",
    "resolved",
  ];

  return (
    <section className="flex min-h-0 flex-col">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-foreground">
          Events{" "}
          <span className="text-subtle">({events.length})</span>
        </h2>
        <div className="flex gap-1">
          {filters.map((value) => (
            <button
              key={value}
              onClick={() => onFilterChange(value)}
              className={`rounded px-2 py-1 text-[11px] capitalize ${
                filter === value
                  ? "bg-accent text-accent-foreground"
                  : "text-muted hover:bg-surface-muted"
              }`}
            >
              {value}
            </button>
          ))}
        </div>
      </div>

      {events.length === 0 ? (
        <p className="rounded-lg border border-dashed border-line p-6 text-center text-xs text-subtle">
          No events yet. Start a stream and step into the restricted zone for
          longer than the dwell threshold.
        </p>
      ) : (
        <ul className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
          {events.map((event) => (
            <EventCard
              key={event.id}
              event={event}
              onUpdated={onUpdated}
              highlighted={event.id === highlightId}
            />
          ))}
        </ul>
      )}
    </section>
  );
}
