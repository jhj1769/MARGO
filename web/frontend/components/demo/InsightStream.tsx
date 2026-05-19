"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  CheckCircle2,
  CircleAlert,
  Compass,
  MessagesSquare,
  Quote,
  Sparkles,
  Wand2
} from "lucide-react";
import type { InsightCards } from "@/lib/api";

/** Natural-language insight cards — what each agent "said" during the last run.
 *  Renders inside the Expert Console below the directive editor so the
 *  operator immediately sees the framework's reaction to their input. */
export function InsightStream({
  insights,
  running
}: {
  insights: InsightCards | null;
  running: boolean;
}) {
  return (
    <div className="surface p-3">
      <div className="flex items-center justify-between">
        <div className="eyebrow flex items-center gap-1.5">
          <MessagesSquare size={11} />
          Agent NL stream
        </div>
        <span className="font-mono text-[10px] text-ash">
          {running ? "live …" : insights ? "settled" : "idle"}
        </span>
      </div>

      <AnimatePresence mode="wait">
        {!insights && !running && (
          <motion.p
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="text-xs italic text-ash mt-2 leading-relaxed"
          >
            Run the pipeline — each MARGO agent will write back here in plain
            English so you can hear them think.
          </motion.p>
        )}

        {!insights && running && <Placeholders key="loading" />}

        {insights && (
          <motion.ul
            key={insights.user_card.summary}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-2 space-y-2"
          >
            {insights.trend_card && (
              <InsightCard
                tone="sage"
                icon={<Compass size={13} />}
                label="Trend Agent"
                title={
                  <>
                    <span className="font-mono text-[10px] text-ash mr-1.5">
                      {insights.trend_card.domain ?? "fashion"} ·{" "}
                      {insights.trend_card.time_window ?? "now"}
                    </span>
                    broadcast
                  </>
                }
                summary={insights.trend_card.summary}
                pills={insights.trend_card.keywords?.slice(0, 6) ?? []}
                quote
              />
            )}

            <InsightCard
              tone="ink"
              icon={<Wand2 size={13} />}
              label="Expert Agent"
              title="Directive issued"
              summary={
                insights.directive_card.natural_language ||
                insights.directive_card.goal
              }
              footer={
                <StructuredRow value={insights.directive_card.structured} />
              }
            />

            <InsightCard
              tone="terracotta"
              icon={<Sparkles size={13} />}
              label={`User Agent · ${insights.user_card.user}`}
              title="Reaction"
              summary={insights.user_card.summary}
              quote
            />

            <InsightCard
              tone={insights.validation_card.passed ? "sage" : "terracotta"}
              icon={
                insights.validation_card.passed ? (
                  <CheckCircle2 size={13} />
                ) : (
                  <CircleAlert size={13} />
                )
              }
              label="Expert Agent · validation"
              title={
                insights.validation_card.passed ? "Passed" : "Refine needed"
              }
              summary={insights.validation_card.summary}
              footer={
                insights.validation_card.violations.length > 0 ? (
                  <ul className="text-[11px] text-terracotta/90 mt-1.5 space-y-1">
                    {insights.validation_card.violations.map((v) => (
                      <li key={v}>· {v}</li>
                    ))}
                  </ul>
                ) : (
                  <span className="font-mono text-[10px] text-ash mt-1.5 inline-flex">
                    compliance {insights.validation_card.compliance.toFixed(2)}
                  </span>
                )
              }
            />
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

function InsightCard({
  tone,
  icon,
  label,
  title,
  summary,
  pills,
  footer,
  quote = false
}: {
  tone: "ink" | "terracotta" | "sage";
  icon?: React.ReactNode;
  label: string;
  title: React.ReactNode;
  summary: string;
  pills?: string[];
  footer?: React.ReactNode;
  quote?: boolean;
}) {
  const toneCls =
    tone === "terracotta"
      ? "border-l-terracotta"
      : tone === "sage"
      ? "border-l-sage"
      : "border-l-ink";
  const dot =
    tone === "terracotta"
      ? "bg-terracotta"
      : tone === "sage"
      ? "bg-sage"
      : "bg-ink";
  return (
    <motion.li
      layout
      initial={{ opacity: 0, x: 8 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
      className={`bg-paper rounded-xl border hairline border-l-4 ${toneCls} p-3`}
    >
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider2 text-ash">
        <span className="inline-flex items-center gap-1.5">
          <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
          {label}
        </span>
        <span className="font-mono">{title}</span>
      </div>
      <p className="text-[12.5px] mt-1.5 leading-relaxed text-ink/85">
        {quote && <Quote size={11} className="inline -mt-0.5 mr-1 text-ash" />}
        {summary}
      </p>
      {pills && pills.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {pills.map((p) => (
            <span key={p} className="chip text-[10px] py-0.5">
              {p}
            </span>
          ))}
        </div>
      )}
      {footer}
    </motion.li>
  );
}

function StructuredRow({ value }: { value: Record<string, unknown> }) {
  const entries = Object.entries(value ?? {}).filter(([, v]) => v !== undefined);
  if (entries.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1 mt-2">
      {entries.map(([k, v]) => (
        <span
          key={k}
          className="inline-flex items-center gap-1 bg-ink/5 rounded-full px-2 py-0.5 text-[10px] font-mono text-ink/80"
        >
          <span className="text-ash">{k}</span>
          <span>{Array.isArray(v) ? v.join(", ") : String(v)}</span>
        </span>
      ))}
    </div>
  );
}

function Placeholders() {
  return (
    <ul className="mt-2 space-y-2">
      {Array.from({ length: 3 }).map((_, i) => (
        <li key={i} className="bg-paper rounded-xl border hairline p-3">
          <div className="h-2.5 w-1/3 bg-sand rounded animate-shimmer bg-[length:200%_100%]" />
          <div className="h-2.5 w-4/5 bg-sand rounded mt-2 animate-shimmer bg-[length:200%_100%]" />
          <div className="h-2.5 w-3/5 bg-sand rounded mt-1.5 animate-shimmer bg-[length:200%_100%]" />
        </li>
      ))}
    </ul>
  );
}
