"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef } from "react";
import { Activity, ArrowRight } from "lucide-react";
import type { TraceEvent } from "@/lib/api";
import { cn } from "@/lib/cn";

const PHASE_COLOR: Record<string, string> = {
  phase1: "border-l-ash",
  phase2: "border-l-ink",
  phase3: "border-l-terracotta",
  phase4: "border-l-sage"
};

const PHASE_LABEL: Record<string, string> = {
  phase1: "Initialization",
  phase2: "Directive",
  phase3: "Reasoning",
  phase4: "Validation"
};

export function TraceTimeline({
  events,
  running
}: {
  events: TraceEvent[];
  running: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [events.length]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-2.5 relative">
      {events.length === 0 && !running && (
        <div className="text-xs text-ash italic py-12 text-center">
          Idle. Run the pipeline to stream agent messages here in real time.
        </div>
      )}

      <AnimatePresence>
        {events.map((e, i) => (
          <motion.div
            key={`${e.ts}-${i}`}
            layout
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className={cn(
              "card p-3 border-l-4",
              PHASE_COLOR[e.phase] ?? "border-l-ash"
            )}
          >
            <div className="flex items-center justify-between text-[10px] uppercase tracking-wider2 text-ash">
              <span>{PHASE_LABEL[e.phase] ?? e.phase}</span>
              <span className="font-mono">#{String(i + 1).padStart(2, "0")}</span>
            </div>

            <div className="flex items-center gap-1.5 mt-1.5">
              <span className="font-mono text-[11px] px-1.5 py-0.5 rounded bg-ink/5">
                {e.sender}
              </span>
              <ArrowRight size={11} className="text-ash" />
              <span className="font-mono text-[11px] px-1.5 py-0.5 rounded bg-paper/70 border hairline text-ash">
                {e.receivers.slice(0, 2).join(", ")}
                {e.receivers.length > 2 ? "…" : ""}
              </span>
            </div>

            <div className="text-sm mt-2 leading-snug text-ink/85">{e.summary}</div>
          </motion.div>
        ))}
      </AnimatePresence>

      {running && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="inline-flex items-center gap-2 text-xs text-ash"
        >
          <Activity size={12} className="text-terracotta animate-pulse" />
          waiting for next message…
        </motion.div>
      )}
    </div>
  );
}
