"use client";

import { motion } from "framer-motion";

export function HeroVisual() {
  return (
    <div className="relative aspect-[4/5] w-full rounded-3xl overflow-hidden border hairline bg-paper">
      {/* Editorial fashion image */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="https://images.unsplash.com/photo-1485518882345-15568b007407?w=1400&q=80"
        alt=""
        className="absolute inset-0 h-full w-full object-cover saturate-[.85]"
      />
      <div className="absolute inset-0 bg-gradient-to-t from-ink/90 via-ink/40 to-transparent" />

      {/* Floating directive chip — written in a real merchandiser's tone */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="absolute top-5 left-5 right-5"
      >
        <div className="rounded-xl bg-paper/95 backdrop-blur p-4 shadow-lg shadow-ink/5">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-wider2 text-ash">
            <span>Merchandiser brief · this week</span>
            <span className="font-mono text-terracotta">issued</span>
          </div>
          <div className="font-display text-[1.05rem] leading-snug mt-1.5 text-ink">
            "Lead with our SS lightweight tailoring — linen blazers and
            breathable trousers.{" "}
            <span className="text-terracotta">
              Don't push prices far above what each shopper usually buys.
            </span>"
          </div>
          <div className="mt-2 flex flex-wrap gap-1 text-[10px] font-mono text-ash">
            <span className="chip py-0.5">season: SS25</span>
            <span className="chip py-0.5">boost: linen tailoring</span>
            <span className="chip py-0.5">respect shopper price band</span>
          </div>
        </div>
      </motion.div>

      {/* Bottom rationale card */}
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.45 }}
        className="absolute bottom-5 left-5 right-5"
      >
        <div className="rounded-xl bg-paper/95 backdrop-blur p-4 shadow-lg shadow-ink/5">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-wider2 text-ash">
            <span>Top-K · Item 01</span>
            <span className="font-mono">score 0.91</span>
          </div>
          <div className="font-display text-lg mt-1">Cream Linen Blazer</div>
          <div className="grid grid-cols-3 gap-1.5 mt-3 text-[10px] uppercase tracking-wider2">
            <Layer color="bg-terracotta">Personal</Layer>
            <Layer color="bg-ink">Directive</Layer>
            <Layer color="bg-sage">Trend</Layer>
          </div>
        </div>
      </motion.div>

      {/* Side decoration: agent message flow */}
      <motion.svg
        viewBox="0 0 80 200"
        className="absolute top-1/2 right-3 -translate-y-1/2 h-44 w-12 opacity-70"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.7 }}
        transition={{ delay: 0.8 }}
      >
        <defs>
          <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#C4664E" stopOpacity="0.0" />
            <stop offset="0.4" stopColor="#C4664E" stopOpacity="0.8" />
            <stop offset="1" stopColor="#C4664E" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[12, 60, 110, 160].map((y, i) => (
          <g key={y}>
            <circle cx="40" cy={y} r="4" fill="#FAF8F4" />
            <text
              x="46"
              y={y + 3}
              fontSize="8"
              fontFamily="ui-monospace"
              fill="#FAF8F4"
            >
              {["expert", "trend", "item", "user"][i]}
            </text>
          </g>
        ))}
        <line
          x1="40"
          y1="12"
          x2="40"
          y2="160"
          stroke="url(#g)"
          strokeWidth="1.5"
          strokeDasharray="3 3"
          className="animate-flow"
        />
      </motion.svg>
    </div>
  );
}

function Layer({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`h-1.5 w-1.5 rounded-full ${color}`} />
      <span className="text-ash">{children}</span>
    </div>
  );
}
