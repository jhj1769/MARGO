// Thin client for the MARGO FastAPI backend.
// All paths funnel through Next.js rewrites in `next.config.mjs`, so on the
// browser side the API root is always `/api/margo`. The WebSocket path is
// `/ws/margo/trace` but rewrites currently only apply to HTTP — we resolve
// the absolute URL ourselves.

export type DirectivePayload = {
  goal: string;
  natural_language: string;
  structured_constraints: Record<string, unknown>;
  issued_at?: number;
  iteration?: number;
};

export type DemoUser = {
  user_id: string;
  name?: string;
  tagline?: string;
  summary?: string;
  history_size?: number;
  price_band?: string;
  avg_price?: number;
  style?: string;
  /** "women" | "men" — only present in engine mode. */
  gender?: string;
};

export type HistoryItem = {
  item_id: string;
  title: string;
  image_url?: string | null;
  category?: string[];
  price?: number | null;
  brand?: string | null;
};

export type UserHistory = {
  user_id: string;
  items: HistoryItem[];
  summary?: string;
  history_size?: number;
};

export type CatalogItem = {
  item_id: string;
  title: string;
  price?: number;
  category?: string[];
  color?: string;
  material?: string;
  image_url?: string;
  brand?: string;
  rating?: number;
  review_count?: number;
  description?: string;
  self_description?: string;
  score?: number;
  rationale?: {
    personal: string;
    directive: string;
    trend: string;
  };
};

export type InsightCards = {
  trend_card: {
    title: string;
    summary: string;
    keywords: string[];
    domain?: string;
    time_window?: string;
  } | null;
  directive_card: {
    title: string;
    goal: string;
    natural_language: string;
    structured: Record<string, unknown>;
  };
  user_card: {
    title: string;
    summary: string;
    user: string;
  };
  validation_card: {
    title: string;
    summary: string;
    passed: boolean;
    compliance: number;
    violations: string[];
  };
};

export type RecommendationResponse = {
  user_id: string;
  directive: DirectivePayload;
  trend: {
    summary: string;
    keywords: string[];
    rising_attributes?: Record<string, string[]>;
  } | null;
  candidate_pool_size: number;
  iterations: number;
  phase4_passed: boolean;
  insights?: InsightCards;
  top_k: CatalogItem[];
};

export type Scenario = {
  id: string;
  title: string;
  blurb: string;
  default_user: string;
  directive: DirectivePayload;
};

export type TraceEvent = {
  phase: "phase1" | "phase2" | "phase3" | "phase4";
  sender: string;
  receivers: string[];
  type: string;
  summary: string;
  payload?: Record<string, unknown>;
  ts: number;
};

const BASE = "/api/margo";

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    ...init,
    cache: "no-store"
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${path} → ${res.status}: ${body || res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => jsonFetch<{ status: string; mode: string }>("/health"),
  users: () => jsonFetch<{ users: DemoUser[] }>("/consumer/users"),
  userHistory: (id: string) =>
    jsonFetch<UserHistory>(`/consumer/users/${encodeURIComponent(id)}/history`),
  scenarios: () => jsonFetch<{ scenarios: Scenario[] }>("/scenarios"),
  recommend: (body: { user_id: string; directive?: DirectivePayload; k?: number }) =>
    jsonFetch<RecommendationResponse>("/consumer/recommend", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  activeDirective: () => jsonFetch<DirectivePayload>("/expert/active-directive"),
  setDirective: (d: DirectivePayload) =>
    jsonFetch<DirectivePayload>("/expert/directive", {
      method: "POST",
      body: JSON.stringify(d)
    })
};

/** Open a WebSocket against the trace endpoint.
 *  Resolves the protocol (ws / wss) based on the current page. */
export function openTraceSocket(): WebSocket {
  if (typeof window === "undefined") {
    throw new Error("openTraceSocket called on the server");
  }
  const isHttps = window.location.protocol === "https:";
  const host = window.location.host;
  // Vite-style fallback: when running the frontend on a different port than
  // the backend, we let the dev user override via env, otherwise hit /ws/margo
  // which is proxied via next.config.mjs rewrites.
  const envWs = process.env.NEXT_PUBLIC_MARGO_WS;
  if (envWs) return new WebSocket(envWs);
  return new WebSocket(`${isHttps ? "wss" : "ws"}://${host}/ws/margo/trace`);
}
