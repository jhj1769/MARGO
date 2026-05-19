"use client";

import { useState } from "react";
import { Plus, Wand2, X } from "lucide-react";
import type { DirectivePayload, InsightCards } from "@/lib/api";
import { InsightStream } from "./InsightStream";

const CHIP_PRESETS: Array<{ key: string; label: string; placeholder: string; type: "number" | "list" }> = [
  { key: "price_diff_pct_max", label: "price gap ≤", placeholder: "30", type: "number" },
  { key: "price_max",          label: "price max",   placeholder: "200", type: "number" },
  { key: "price_min",          label: "price min",   placeholder: "50",  type: "number" },
  { key: "boost_category",     label: "boost",       placeholder: "trench-coat, blazer", type: "list" },
  { key: "forbid_category",    label: "forbid",      placeholder: "lingerie", type: "list" }
];

export function ExpertConsole({
  directive,
  onChange,
  onRun,
  running,
  k,
  onK,
  insights
}: {
  directive: DirectivePayload;
  onChange: (d: DirectivePayload) => void;
  onRun: () => void;
  running: boolean;
  k: number;
  onK: (k: number) => void;
  insights: InsightCards | null;
}) {
  const sc = directive.structured_constraints ?? {};
  const [adding, setAdding] = useState<string | null>(null);
  const [draft, setDraft] = useState<string>("");

  function updateConstraint(key: string, raw: string, type: "number" | "list") {
    const next: any = { ...sc };
    if (raw.trim() === "") {
      delete next[key];
    } else if (type === "number") {
      const n = Number(raw);
      if (!Number.isNaN(n)) next[key] = n;
    } else {
      next[key] = raw.split(",").map((s) => s.trim()).filter(Boolean);
    }
    onChange({ ...directive, structured_constraints: next });
  }

  function removeConstraint(key: string) {
    const next: any = { ...sc };
    delete next[key];
    onChange({ ...directive, structured_constraints: next });
  }

  return (
    <div className="flex flex-col gap-4 p-4 overflow-y-auto">
      {/* Goal */}
      <div>
        <label className="eyebrow">Goal</label>
        <input
          className="mt-1 w-full rounded-lg border hairline bg-paper px-3 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-ink/15"
          placeholder="Casual → formal upsell"
          value={directive.goal ?? ""}
          onChange={(e) => onChange({ ...directive, goal: e.target.value })}
        />
      </div>

      {/* Natural Language */}
      <div>
        <label className="eyebrow flex items-center justify-between">
          <span>Directive · natural language</span>
          <span className="font-mono text-[10px]">phase 2</span>
        </label>
        <textarea
          className="mt-1 w-full min-h-[110px] rounded-lg border hairline bg-paper px-3 py-2 text-sm leading-relaxed focus:outline-none focus:ring-2 focus:ring-ink/15"
          placeholder="Move casual shoppers toward their first formal capsule piece. Keep the price jump comfortable. Boost trench coats and tailored outerwear."
          value={directive.natural_language ?? ""}
          onChange={(e) =>
            onChange({ ...directive, natural_language: e.target.value })
          }
        />
      </div>

      {/* Constraints */}
      <div>
        <div className="flex items-baseline justify-between">
          <label className="eyebrow">Structured constraints</label>
          <span className="text-[10px] text-ash font-mono">machine-checkable</span>
        </div>

        <div className="mt-2 flex flex-wrap gap-1.5">
          {Object.entries(sc).map(([key, val]) => {
            const preset = CHIP_PRESETS.find((p) => p.key === key);
            const display = Array.isArray(val) ? val.join(", ") : String(val);
            return (
              <span
                key={key}
                className="inline-flex items-center gap-2 rounded-full bg-ink text-paper text-xs px-3 py-1.5"
              >
                <span className="font-mono opacity-70">
                  {preset?.label ?? key}
                </span>
                <span className="font-medium">{display}</span>
                <button
                  onClick={() => removeConstraint(key)}
                  className="opacity-70 hover:opacity-100"
                >
                  <X size={11} />
                </button>
              </span>
            );
          })}

          {/* Add new */}
          {adding ? (
            <span className="inline-flex items-center gap-1 bg-paper border hairline rounded-full px-2 py-1 text-xs">
              <span className="font-mono text-ash">
                {CHIP_PRESETS.find((p) => p.key === adding)?.label}
              </span>
              <input
                autoFocus
                className="w-28 bg-transparent focus:outline-none px-1"
                placeholder={CHIP_PRESETS.find((p) => p.key === adding)?.placeholder}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && adding) {
                    const preset = CHIP_PRESETS.find((p) => p.key === adding);
                    updateConstraint(adding, draft, preset?.type ?? "list");
                    setAdding(null);
                    setDraft("");
                  }
                  if (e.key === "Escape") {
                    setAdding(null);
                    setDraft("");
                  }
                }}
              />
              <button onClick={() => { setAdding(null); setDraft(""); }}>
                <X size={11} />
              </button>
            </span>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {CHIP_PRESETS.filter((p) => !(p.key in sc)).map((p) => (
                <button
                  key={p.key}
                  onClick={() => setAdding(p.key)}
                  className="inline-flex items-center gap-1.5 rounded-full border border-dashed border-ink/20 px-2.5 py-1 text-xs text-ink/70 hover:bg-ink/5"
                >
                  <Plus size={11} />
                  {p.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* NL stream — Trend / Expert / User / Validation insights */}
      <InsightStream insights={insights} running={running} />

      {/* K + Run */}
      <div className="surface p-3 flex items-center justify-between">
        <label className="text-xs text-ink/70 flex items-center gap-2">
          Top-K
          <input
            type="number"
            min={2}
            max={20}
            value={k}
            onChange={(e) => onK(Math.max(2, Math.min(20, Number(e.target.value) || 6)))}
            className="w-14 rounded-md border hairline bg-paper px-2 py-1 text-sm font-mono"
          />
        </label>
        <button onClick={onRun} disabled={running} className="btn-terracotta">
          <Wand2 size={14} />
          {running ? "Running…" : "Run pipeline"}
        </button>
      </div>
    </div>
  );
}
