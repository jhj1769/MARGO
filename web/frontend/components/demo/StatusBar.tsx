"use client";

import { AlertCircle, Activity, CheckCircle2, Loader2, PlayCircle, RefreshCw, Cpu } from "lucide-react";
import type { DirectivePayload } from "@/lib/api";

export function StatusBar({
  mode,
  directive,
  running,
  onRun,
  traceCount,
  passed,
  iterations,
  pool,
  error
}: {
  mode: "engine" | "mock" | "offline";
  directive: DirectivePayload;
  running: boolean;
  onRun: () => void;
  traceCount: number;
  passed?: boolean;
  iterations?: number;
  pool?: number;
  error?: string | null;
}) {
  return (
    <div className="surface px-4 py-3 flex flex-wrap items-center gap-3">
      <BadgeMode mode={mode} />

      <span className="hairline border-l h-5" />

      <Metric label="iter"      value={iterations ?? "—"} />
      <Metric label="pool"      value={pool ?? "—"} />
      <Metric label="trace"     value={traceCount} />
      <Metric
        label="compliance_score"
        value={passed === undefined ? "—" : passed ? "pass" : "refine"}
        valueClass={
          passed === undefined
            ? "text-ash"
            : passed
            ? "text-sage"
            : "text-terracotta"
        }
      />

      <div className="ml-auto flex items-center gap-2">
        {error && (
          <span className="chip border-terracotta/40 text-terracotta">
            <AlertCircle size={12} /> {error}
          </span>
        )}
        <button
          onClick={onRun}
          disabled={running || !directive.natural_language}
          className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {running ? (
            <>
              <Loader2 size={14} className="animate-spin" />
              Running MARGO…
            </>
          ) : (
            <>
              <PlayCircle size={16} />
              Run pipeline
            </>
          )}
        </button>
      </div>
    </div>
  );
}

function BadgeMode({ mode }: { mode: "engine" | "mock" | "offline" }) {
  if (mode === "engine") {
    return (
      <span className="chip border-sage/40 text-sage">
        <Cpu size={12} /> engine · live LLM
      </span>
    );
  }
  if (mode === "mock") {
    return (
      <span className="chip">
        <Activity size={12} className="text-terracotta" /> mock backend
      </span>
    );
  }
  return (
    <span className="chip border-terracotta/40 text-terracotta">
      <RefreshCw size={12} /> backend offline
    </span>
  );
}

function Metric({
  label,
  value,
  valueClass
}: {
  label: string;
  value: number | string | undefined;
  valueClass?: string;
}) {
  return (
    <span className="inline-flex items-baseline gap-1.5 text-xs">
      <span className="uppercase tracking-wider2 text-ash">{label}</span>
      <span className={`font-mono ${valueClass ?? "text-ink"}`}>
        {value ?? "—"}
      </span>
    </span>
  );
}
