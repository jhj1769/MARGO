"use client";

import { motion } from "framer-motion";

const ITEMS = [
  {
    id: "f_001",
    title: "Beige Midi Trench Coat",
    price: "₩168k",
    image:
      "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=600&q=80",
    score: 0.91,
    rationale: {
      personal:
        "Adjacent to your usual cotton-blend outerwear, fits the ₩50–100k expansion you've shown.",
      directive:
        "Honors the operator's 'casual → formal' goal. Price gap +24% (≤30% cap).",
      trend: "Aligned with SS26 — earth-toned, longer silhouettes are dominant."
    }
  },
  {
    id: "f_005",
    title: "Penny Loafers · Burgundy",
    price: "₩142k",
    image:
      "https://images.unsplash.com/photo-1614252369475-531eba835eb1?w=600&q=80",
    score: 0.86,
    rationale: {
      personal:
        "Pairs naturally with the chinos you bought last quarter — leather is new for you, but bridged.",
      directive:
        "Within boost category 'formal'. Soft entry: pairs upward without committing to a suit.",
      trend: "Burgundy is part of the rising tonal palette for spring tailoring."
    }
  }
];

export function RationalePreview() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
      {ITEMS.map((it, i) => (
        <motion.article
          key={it.id}
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-20% 0px" }}
          transition={{ duration: 0.5, delay: 0.05 * i }}
          className="rounded-2xl overflow-hidden border border-paper/15 bg-ink/40 backdrop-blur"
        >
          <div className="relative aspect-[5/4]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={it.image}
              alt={it.title}
              className="absolute inset-0 h-full w-full object-cover saturate-[.85]"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-ink to-transparent" />
            <div className="absolute bottom-4 left-4 right-4 flex items-end justify-between">
              <div>
                <div className="text-[10px] uppercase tracking-wider2 text-paper/60">
                  Top-K · {String(i + 1).padStart(2, "0")}
                </div>
                <div className="font-display text-2xl text-paper tracking-editorial">
                  {it.title}
                </div>
              </div>
              <div className="text-right">
                <div className="text-[10px] uppercase tracking-wider2 text-paper/60">
                  price
                </div>
                <div className="font-mono text-paper">{it.price}</div>
                <div className="mt-1.5 inline-flex items-center gap-1.5 chip-solid bg-terracotta">
                  <span className="font-mono text-[10px]">{it.score.toFixed(2)}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 divide-y divide-paper/10">
            <Row label="Personal" tone="terracotta" text={it.rationale.personal} />
            <Row label="Directive" tone="gold" text={it.rationale.directive} />
            <Row label="Trend" tone="sage" text={it.rationale.trend} />
          </div>
        </motion.article>
      ))}
    </div>
  );
}

function Row({
  label,
  text,
  tone
}: {
  label: string;
  text: string;
  tone: "terracotta" | "gold" | "sage";
}) {
  const toneClass =
    tone === "terracotta"
      ? "bg-terracotta"
      : tone === "gold"
      ? "bg-gold"
      : "bg-sage";
  return (
    <div className="px-4 py-3 flex items-start gap-3">
      <span
        className={`mt-1 h-1.5 w-1.5 rounded-full shrink-0 ${toneClass}`}
        aria-hidden
      />
      <div>
        <div className="text-[10px] uppercase tracking-wider2 text-paper/60">
          {label}
        </div>
        <div className="text-sm text-paper/90 leading-relaxed mt-0.5">{text}</div>
      </div>
    </div>
  );
}
