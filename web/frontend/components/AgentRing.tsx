"use client";

import { motion } from "framer-motion";

/** A round diagram with the four MARGO agents arranged around a central
 *  message bus. Pure SVG — no canvas, no external deps. */
export function AgentRing() {
  const R = 120;
  const cx = 180;
  const cy = 180;

  const agents = [
    { id: "user",   label: "User",   angle: -90, color: "#C4664E" },
    { id: "item",   label: "Item",   angle: 0,   color: "#0F1B2D" },
    { id: "expert", label: "Expert", angle: 90,  color: "#B68F4F" },
    { id: "trend",  label: "Trend",  angle: 180, color: "#7F8B5A" }
  ];

  return (
    <div className="surface aspect-square p-4 relative">
      <svg viewBox="0 0 360 360" className="w-full h-full">
        {/* outer hairline ring */}
        <circle cx={cx} cy={cy} r={R + 30} fill="none" stroke="#0F1B2D" strokeOpacity="0.06" />
        <circle cx={cx} cy={cy} r={R - 30} fill="none" stroke="#0F1B2D" strokeOpacity="0.06" />

        {/* connecting lines between agents and center */}
        {agents.map((a) => {
          const rad = (a.angle * Math.PI) / 180;
          const x = cx + R * Math.cos(rad);
          const y = cy + R * Math.sin(rad);
          return (
            <g key={a.id}>
              <line
                x1={cx}
                y1={cy}
                x2={x}
                y2={y}
                stroke="#0F1B2D"
                strokeOpacity="0.15"
                strokeDasharray="2 5"
              />
            </g>
          );
        })}

        {/* center bus */}
        <circle cx={cx} cy={cy} r="44" fill="#FAF8F4" stroke="#0F1B2D" />
        <text
          x={cx}
          y={cy - 4}
          textAnchor="middle"
          fontSize="10"
          fontFamily="ui-monospace"
          fill="#7B8597"
        >
          MESSAGE
        </text>
        <text
          x={cx}
          y={cy + 10}
          textAnchor="middle"
          fontSize="10"
          fontFamily="ui-monospace"
          fill="#7B8597"
        >
          BUS
        </text>

        {/* agent nodes */}
        {agents.map((a, idx) => {
          const rad = (a.angle * Math.PI) / 180;
          const x = cx + R * Math.cos(rad);
          const y = cy + R * Math.sin(rad);
          return (
            <motion.g
              key={a.id}
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 * idx, duration: 0.4 }}
            >
              <circle cx={x} cy={y} r="36" fill={a.color} />
              <text
                x={x}
                y={y - 4}
                textAnchor="middle"
                fontFamily="Fraunces, serif"
                fontSize="17"
                fill="#FAF8F4"
              >
                {a.label}
              </text>
              <text
                x={x}
                y={y + 12}
                textAnchor="middle"
                fontFamily="ui-monospace"
                fontSize="8"
                fill="#FAF8F4"
                opacity="0.7"
              >
                agent
              </text>
            </motion.g>
          );
        })}

        {/* animated pulse */}
        <motion.circle
          cx={cx}
          cy={cy}
          r="55"
          fill="none"
          stroke="#C4664E"
          initial={{ opacity: 0.6, r: 50 }}
          animate={{ opacity: 0, r: 90 }}
          transition={{ duration: 2.2, repeat: Infinity, ease: "easeOut" }}
        />
      </svg>

      <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between text-[10px] uppercase tracking-wider2 text-ash">
        <span>4-Stakeholder Collaboration</span>
        <span className="font-mono">v0.1 · margo</span>
      </div>
    </div>
  );
}
