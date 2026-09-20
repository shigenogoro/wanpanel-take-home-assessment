"use client";

import type { ConnectionState } from "@/hooks/useEventChannel";
import type { RuleConfig, SystemStatus } from "@/lib/types";

interface Props {
  status: SystemStatus | null;
  config: RuleConfig | null;
  channel: ConnectionState;
  unreachable: string | null;
}

function Card({
  label,
  value,
  tone = "normal",
  hint,
}: {
  label: string;
  value: string;
  tone?: "normal" | "good" | "bad";
  hint?: string;
}) {
  const toneClass =
    tone === "good"
      ? "text-ok"
      : tone === "bad"
        ? "text-danger"
        : "text-foreground";
  return (
    <div className="rounded-lg border border-line bg-surface p-3">
      <div className="text-[11px] uppercase tracking-wide text-muted">
        {label}
      </div>
      <div className={`mt-1 truncate text-sm font-semibold ${toneClass}`} title={value}>
        {value}
      </div>
      {hint && <div className="mt-0.5 text-[11px] text-subtle">{hint}</div>}
    </div>
  );
}

export function StatusPanel({ status, config, channel, unreachable }: Props) {
  if (unreachable) {
    return (
      <div className="rounded-lg border border-red-500 bg-red-500/10 p-4 text-sm text-danger">
        <p className="font-semibold">Backend unreachable</p>
        <p className="mt-1">{unreachable}</p>
        <p className="mt-2 text-xs">
          Start it with{" "}
          <code className="rounded bg-foreground/10 px-1">
            uvicorn app.main:app --port 8000
          </code>{" "}
          from the <code className="rounded bg-foreground/10 px-1">backend/</code>{" "}
          directory.
        </p>
      </div>
    );
  }

  const model = status?.model;
  const metrics = status?.metrics;

  return (
    <div className="space-y-3">
      {model && !model.loaded && (
        <div className="rounded-lg border border-amber-500 bg-amber-500/10 p-4 text-sm text-warn">
          <p className="font-semibold">Model not loaded — inference disabled</p>
          <p className="mt-1">{model.error}</p>
        </div>
      )}

      {status && !status.database_connected && (
        <div className="rounded-lg border border-amber-500 bg-amber-500/10 p-4 text-sm text-warn">
          <p className="font-semibold">Database unreachable</p>
          <p className="mt-1">
            Events cannot be saved. Check <code>DATABASE_URL</code> in{" "}
            <code>backend/.env</code>.
          </p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2 lg:grid-cols-3">
        <Card
          label="Model"
          value={model ? model.name : "—"}
          hint={model?.version}
        />
        <Card
          label="Inference"
          value={model?.loaded ? "ready" : "unavailable"}
          tone={model?.loaded ? "good" : "bad"}
          hint={model?.provider}
        />
        <Card
          label="Database"
          value={status?.database_connected ? "connected" : "down"}
          tone={status?.database_connected ? "good" : "bad"}
        />
        <Card
          label="Camera source"
          value={status?.active_source ?? "idle"}
          hint={`${status?.inference_clients ?? 0} stream(s)`}
        />
        <Card
          label="Throughput"
          value={`${metrics?.fps?.toFixed(1) ?? "0.0"} fps`}
          hint={`${metrics?.frames_processed ?? 0} frames processed`}
        />
        <Card
          label="Model latency"
          value={`p50 ${metrics?.inference_ms_p50?.toFixed(0) ?? 0} ms`}
          hint={`p95 ${metrics?.inference_ms_p95?.toFixed(0) ?? 0} ms · mean ${
            metrics?.inference_ms_mean?.toFixed(0) ?? 0
          } ms`}
        />
        <Card
          label="Event channel"
          value={channel}
          tone={channel === "open" ? "good" : "bad"}
          hint="live push, no polling"
        />
        <Card
          label="Zone"
          value={status?.zone?.name ?? "none"}
          hint={`${status?.zone?.polygon.length ?? 0} vertices`}
        />
        <Card
          label="Trigger rule"
          value={config ? `${config.dwell_seconds}s dwell` : "—"}
          hint={
            config
              ? `conf ≥ ${config.conf_threshold} · cooldown ${config.track_cooldown_seconds}s`
              : undefined
          }
        />
      </div>

      {model && (
        <p className="text-[11px] text-subtle">
          {model.name} {model.version} · {model.license} ·{" "}
          <a
            href={model.source}
            target="_blank"
            rel="noreferrer"
            className="underline hover:text-foreground"
          >
            model source
          </a>
        </p>
      )}
    </div>
  );
}
