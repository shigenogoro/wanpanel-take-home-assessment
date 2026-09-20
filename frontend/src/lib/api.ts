import type {
  EventList,
  EventStatus,
  RuleConfig,
  SafetyEvent,
  SystemStatus,
  Zone,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export const WS_BASE = API_BASE.replace(/^http/, "ws");

/** Thrown for any non-2xx response, carrying the backend's `detail` message. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    // A network-level failure means the backend is not reachable at all --
    // worth saying plainly rather than surfacing a bare TypeError.
    throw new ApiError(`Cannot reach the API at ${API_BASE}`, 0);
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body; keep the status text */
    }
    throw new ApiError(detail, response.status);
  }

  return response.status === 204 ? (undefined as T) : response.json();
}

export const api = {
  status: () => request<SystemStatus>("/api/status"),
  config: () => request<RuleConfig>("/api/config"),

  listEvents: (params: { status?: EventStatus; limit?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.status) query.set("status", params.status);
    query.set("limit", String(params.limit ?? 50));
    return request<EventList>(`/api/events?${query}`);
  },

  setEventStatus: (id: string, status: EventStatus) =>
    request<SafetyEvent>(`/api/events/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    }),

  createTestEvent: (label: string) =>
    request<SafetyEvent>("/api/events/test", {
      method: "POST",
      body: JSON.stringify({ label }),
    }),

  activeZone: () => request<Zone>("/api/zones/active"),

  updateZone: (id: number, polygon: [number, number][]) =>
    request<Zone>(`/api/zones/${id}`, {
      method: "PUT",
      body: JSON.stringify({ polygon }),
    }),

  snapshotUrl: (id: string) => `${API_BASE}/api/events/${id}/snapshot`,
};
