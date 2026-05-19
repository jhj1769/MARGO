"use client";

import { Sparkles } from "lucide-react";
import type { Scenario } from "@/lib/api";

export function ScenarioBar({
  scenarios,
  onPick
}: {
  scenarios: Scenario[];
  onPick: (s: Scenario) => void;
}) {
  if (scenarios.length === 0) return null;
  return (
    <div>
      <div className="flex items-center gap-2 mb-2 text-xs text-ash">
        <Sparkles size={12} className="text-terracotta" />
        <span className="uppercase tracking-wider2">Operator scenarios</span>
        <span className="text-ash/70">— click a preset to auto-fill the directive, or skip and write your own.</span>
      </div>
      <div className="flex flex-wrap items-stretch gap-3">
        {scenarios.map((s) => (
          <button
            key={s.id}
            onClick={() => onPick(s)}
            className="card group p-4 flex-1 min-w-[240px] text-left hover:border-ink/25 hover:-translate-y-0.5 transition"
          >
            <div className="font-display tracking-editorial text-lg">
              {s.title}
            </div>
            <div className="text-[12px] text-ink/70 mt-1.5 leading-relaxed">
              {s.blurb}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
