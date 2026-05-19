"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  openTraceSocket,
  type CatalogItem,
  type DemoUser,
  type DirectivePayload,
  type RecommendationResponse,
  type Scenario,
  type TraceEvent
} from "@/lib/api";
import { ConsumerView } from "./ConsumerView";
import { ExpertConsole } from "./ExpertConsole";
import { TraceTimeline } from "./TraceTimeline";
import { UserPicker } from "./UserPicker";
import { ScenarioBar } from "./ScenarioBar";
import { StatusBar } from "./StatusBar";

const DEFAULT_DIRECTIVE: DirectivePayload = {
  goal: "",
  natural_language: "",
  structured_constraints: {}
};

export function DemoWorkbench() {
  // ─── server data ───────────────────────────────────────────────────────
  const [users, setUsers] = useState<DemoUser[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [serverMode, setServerMode] = useState<"engine" | "mock" | "offline">("offline");

  // ─── editable state ────────────────────────────────────────────────────
  const [activeUser, setActiveUser] = useState<string>("");
  const [directive, setDirective] = useState<DirectivePayload>(DEFAULT_DIRECTIVE);
  const [k, setK] = useState(6);

  // ─── run state ─────────────────────────────────────────────────────────
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // ─── bootstrap ─────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const [h, u, s] = await Promise.all([
          api.health(),
          api.users(),
          api.scenarios()
        ]);
        setServerMode(h.mode === "engine" ? "engine" : "mock");
        setUsers(u.users);
        setScenarios(s.scenarios);
        if (s.scenarios[0]) {
          setDirective(s.scenarios[0].directive);
        }
        // Default to first real user from roster (not mock u_xxx)
        const firstReal = u.users.find((x: any) => !x.user_id.startsWith("u_"));
        if (firstReal) {
          setActiveUser(firstReal.user_id);
        } else if (u.users[0]) {
          setActiveUser(u.users[0].user_id);
        }
      } catch (e) {
        console.error(e);
        setServerMode("offline");
        setError(
          "Backend unreachable. Run `uvicorn web.backend.main:app --port 8001` and refresh."
        );
      }
    })();
  }, []);

  // ─── run ───────────────────────────────────────────────────────────────
  const run = useCallback(() => {
    if (!activeUser || running) return;
    setRunning(true);
    setError(null);
    setTraceEvents([]);
    setResult(null);

    try {
      const ws = openTraceSocket();
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            action: "recommend",
            user_id: activeUser,
            directive,
            k
          })
        );
      };
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.event === "trace") {
            setTraceEvents((prev) => [...prev, msg.data as TraceEvent]);
          } else if (msg.event === "result") {
            setResult(msg.data as RecommendationResponse);
            setRunning(false);
            ws.close();
          } else if (msg.event === "error") {
            setError(msg.message ?? "trace error");
            setRunning(false);
            ws.close();
          }
        } catch (err) {
          console.error("ws parse", err);
        }
      };
      ws.onerror = async () => {
        // Fall back to plain HTTP — the WS rewrite isn't supported by
        // every Next dev server; the consumer endpoint still works.
        try {
          const r = await api.recommend({ user_id: activeUser, directive, k });
          setResult(r);
        } catch (e: any) {
          setError(e?.message ?? "request failed");
        } finally {
          setRunning(false);
        }
      };
      ws.onclose = () => {
        setRunning(false);
      };
    } catch (e: any) {
      setError(e?.message ?? "could not open trace");
      setRunning(false);
    }
  }, [activeUser, directive, k, running]);

  // close WS on unmount
  useEffect(() => () => wsRef.current?.close(), []);

  // ─── derived ───────────────────────────────────────────────────────────
  const currentUser = useMemo(
    () => users.find((u) => u.user_id === activeUser) ?? null,
    [users, activeUser]
  );
  const items: CatalogItem[] = result?.top_k ?? [];

  return (
    <div className="container-wide pb-12 space-y-4">
      <StatusBar
        mode={serverMode}
        directive={directive}
        running={running}
        onRun={run}
        traceCount={traceEvents.length}
        passed={result?.phase4_passed}
        iterations={result?.iterations}
        pool={result?.candidate_pool_size}
        error={error}
      />

      <ScenarioBar
        scenarios={scenarios}
        onPick={(s) => {
          setDirective(s.directive);
          // Only switch user if it's a real Amazon user (not mock u_xxx)
          if (s.default_user && !s.default_user.startsWith("u_")) {
            setActiveUser(s.default_user);
          }
        }}
      />

      <UserPicker
        users={users}
        active={activeUser}
        onPick={setActiveUser}
        currentUser={currentUser}
      />

      <div className="grid grid-cols-12 gap-4 min-h-[640px]">
        {/* Operator Directive — input (Phase 2) */}
        <section className="col-span-12 lg:col-span-4 surface flex flex-col ring-1 ring-terracotta/15 shadow-sm">
          <div className="px-4 py-3 border-b hairline flex items-center justify-between bg-terracotta/[0.04]">
            <div>
              <div className="eyebrow flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-terracotta" />
                Step 2 · Operator Directive
              </div>
              <div className="font-display tracking-editorial text-lg">
                What should we recommend?
              </div>
            </div>
            <span className="chip">Input</span>
          </div>
          <ExpertConsole
            directive={directive}
            onChange={setDirective}
            onRun={run}
            running={running}
            k={k}
            onK={setK}
            insights={result?.insights ?? null}
          />
        </section>

        {/* Top-K Ranked List — main output (Phase 3) */}
        <section className="col-span-12 lg:col-span-5 surface flex flex-col">
          <div className="px-4 py-3 border-b hairline flex items-center justify-between">
            <div>
              <div className="eyebrow">Step 3 · Output</div>
              <div className="font-display tracking-editorial text-lg">
                Top-K Ranked List
              </div>
            </div>
            <span className="chip">
              {currentUser?.name ?? activeUser} · top-{k}
            </span>
          </div>
          <ConsumerView
            items={items}
            running={running}
            currentUser={currentUser}
          />
        </section>

        {/* Agent activity — Multi-Agent Reasoning trace (Phase 3) */}
        <section className="col-span-12 lg:col-span-3 surface flex flex-col">
          <div className="px-4 py-3 border-b hairline">
            <div className="eyebrow">Multi-Agent Reasoning</div>
            <div className="font-display tracking-editorial text-lg">
              Agent activity
            </div>
          </div>
          <TraceTimeline events={traceEvents} running={running} />
        </section>
      </div>
    </div>
  );
}
