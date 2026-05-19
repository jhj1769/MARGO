"use client";

import { motion } from "framer-motion";

/** 4-Phase pipeline diagram with a refine-loop arrow. Designed wide so it
 *  reads like a printed figure in a paper. */
export function PhaseDiagram() {
  const phases = [
    { id: "01", label: "Agent Initialization", small: "offline", color: "#7B8597" },
    { id: "02", label: "Directive Generation", small: "expert + trend", color: "#0F1B2D" },
    { id: "03", label: "Multi-Agent Reasoning", small: "retrieve · describe · rank", color: "#C4664E" },
    { id: "04", label: "Validation & Refinement", small: "verify · refine", color: "#7F8B5A" }
  ];

  return (
    <div className="surface p-6 lg:p-10 relative">
      <div className="absolute top-4 left-6 eyebrow">Lifecycle</div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-10">
        {phases.map((p, i) => (
          <motion.div
            key={p.id}
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: i * 0.08 }}
            className="relative card p-5"
          >
            <div className="flex items-center justify-between">
              <span
                className="font-mono text-xs px-2 py-1 rounded-full"
                style={{ background: p.color, color: "#FAF8F4" }}
              >
                phase {p.id}
              </span>
              <span className="text-[10px] uppercase tracking-wider2 text-ash">
                {p.small}
              </span>
            </div>
            <h3 className="font-display text-xl tracking-editorial mt-3 leading-tight">
              {p.label}
            </h3>
            <div className="mt-4 grid grid-cols-4 gap-1">
              {Array.from({ length: 4 }).map((_, idx) => (
                <span
                  key={idx}
                  className={`h-1 rounded ${
                    idx <= i ? "" : "bg-ink/10"
                  }`}
                  style={idx <= i ? { background: p.color } : {}}
                />
              ))}
            </div>
            {i < phases.length - 1 && (
              <div className="hidden md:block absolute right-[-30px] top-1/2 -translate-y-1/2 text-ash">
                →
              </div>
            )}
          </motion.div>
        ))}
      </div>

      {/* Detailed graph */}
      <div className="mt-12 grid grid-cols-12 gap-4">
        {/* Inputs */}
        <div className="col-span-12 md:col-span-3 space-y-3">
          <DiagramNode label="Operator Directive"   sub="natural language" tone="terracotta" />
          <DiagramNode label="Web Evidence"     sub="trend agent input" tone="sage" />
          <DiagramNode label="User Logs"        sub="history (offline)" />
          <DiagramNode label="Item Catalog"     sub="attributes (offline)" />
        </div>

        {/* Center: agents */}
        <div className="col-span-12 md:col-span-6 surface p-4 md:p-6 relative">
          <div className="eyebrow mb-3">Multi-Agent Reasoning</div>
          <div className="grid grid-cols-2 gap-3">
            <AgentBlock title="Expert Agent" body="Issues directive · validates output · refines policy" />
            <AgentBlock title="Trend Agent"  body="Interprets external context · broadcasts to all" />
            <AgentBlock title="User Agent"   body="Evaluates candidates · 3-layer rationale" />
            <AgentBlock title="Item Agent"   body="Self-describes under directive + trend" />
          </div>
          <div className="mt-4 rounded-xl border hairline bg-paper p-3 flex items-center justify-between text-xs">
            <span className="font-mono text-ash">BGE-M3 Retriever</span>
            <span className="text-ink/70">directive-aware candidate pool</span>
          </div>

          <div className="absolute -right-3 top-3 text-[10px] tracking-wider2 uppercase text-ash hidden md:block">
            phase 3
          </div>
        </div>

        {/* Output */}
        <div className="col-span-12 md:col-span-3 space-y-3">
          <DiagramNode tone="ink" label="Top-K Ranked List" sub="recommendation output" />
          <DiagramNode tone="terracotta" label="3-Layer Rationale" sub="personal · directive · trend" />
          <DiagramNode tone="sage" label="Compliance Report" sub="structured + NL verdict" />
          <div className="rounded-xl border hairline border-dashed p-3 text-xs text-ash">
            ↺ <span className="font-mono">refine_directive</span> — when compliance &lt; 0.85, the
            Expert Agent rewrites the directive and Phase 2 re-runs (max 3 iterations).
          </div>
        </div>
      </div>
    </div>
  );
}

function DiagramNode({
  label,
  sub,
  tone
}: {
  label: string;
  sub?: string;
  tone?: "terracotta" | "sage" | "ink";
}) {
  const toneClass =
    tone === "terracotta"
      ? "border-l-terracotta"
      : tone === "sage"
      ? "border-l-sage"
      : tone === "ink"
      ? "border-l-ink"
      : "border-l-ash";
  return (
    <div className={`card p-3.5 border-l-4 ${toneClass}`}>
      <div className="text-sm font-medium">{label}</div>
      {sub && <div className="text-[11px] text-ash mt-0.5">{sub}</div>}
    </div>
  );
}

function AgentBlock({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-xl border hairline bg-paper p-3">
      <div className="font-display text-sm tracking-editorial">{title}</div>
      <div className="text-[11px] text-ash mt-1 leading-relaxed">{body}</div>
    </div>
  );
}
