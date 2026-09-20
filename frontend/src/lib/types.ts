/** Mirrors the Pydantic schemas in backend/app/schemas.py. */

export type EventStatus = "open" | "acknowledged" | "resolved";

export interface Detection {
  track_id: number;
  label: string;
  confidence: number;
  /** Normalised [x1, y1, x2, y2]. */
  bbox: [number, number, number, number];
  /** Normalised [x, y] anchor the zone rule tests. */
  foot: [number, number];
  in_zone: boolean;
}

export interface ZoneState {
  zone_id: number | null;
  occupied_track_ids: number[];
  dwell_seconds: number;
}

export interface InferenceResult {
  type: "inference";
  frame_id: number;
  ts: string;
  /** Model time only; excludes transport. */
  latency_ms: number;
  fps: number;
  detections: Detection[];
  zone_state: ZoneState;
  events_fired: string[];
}

export interface WsError {
  type: "error";
  code: string;
  message: string;
}

export type InferenceMessage = InferenceResult | WsError;

export interface SafetyEvent {
  id: string;
  event_time: string;
  created_at: string;
  type: string;
  label: string;
  confidence: number;
  source: string;
  model_name: string;
  model_version: string;
  status: EventStatus;
  zone_id: number | null;
  track_id: number | null;
  bbox: [number, number, number, number] | null;
  reason: Record<string, unknown> | null;
  has_snapshot: boolean;
  acknowledged_at: string | null;
  resolved_at: string | null;
}

export interface EventList {
  items: SafetyEvent[];
  total: number;
  limit: number;
  offset: number;
}

export interface Zone {
  id: number;
  name: string;
  polygon: [number, number][];
  active: boolean;
  updated_at: string;
}

export interface ModelStatus {
  name: string;
  version: string;
  license: string;
  source: string;
  loaded: boolean;
  provider: string;
  error: string | null;
}

export interface SystemStatus {
  model: ModelStatus;
  database_connected: boolean;
  active_source: string | null;
  inference_clients: number;
  dashboard_clients: number;
  metrics: {
    frames_processed: number;
    frames_rejected: number;
    inference_ms_mean: number;
    inference_ms_p50: number;
    inference_ms_p95: number;
    fps: number;
    uptime_seconds: number;
    seconds_since_last_frame: number | null;
  };
  zone: Zone | null;
}

export interface RuleConfig {
  conf_threshold: number;
  dwell_seconds: number;
  hysteresis_grace_seconds: number;
  track_cooldown_seconds: number;
  zone_cooldown_seconds: number;
  absence_seconds: number;
  absence_cooldown_seconds: number;
  track_iou_threshold: number;
  track_max_age_frames: number;
  max_frame_bytes: number;
}

/** Broadcast on /ws/events. */
export type EventChannelMessage =
  | { type: "event.created"; payload: SafetyEvent }
  | { type: "event.updated"; payload: SafetyEvent };
