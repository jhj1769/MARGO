"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, Heart, ShoppingBag, Sparkles, Star } from "lucide-react";
import { useState } from "react";
import type { CatalogItem, DemoUser } from "@/lib/api";

export function ConsumerView({
  items,
  running,
  currentUser
}: {
  items: CatalogItem[];
  running: boolean;
  currentUser: DemoUser | null;
}) {
  const featured = items[0];
  const rest = items.slice(1);

  return (
    <div className="flex flex-col overflow-y-auto">
      {/* Storefront banner */}
      <div className="px-4 pt-4">
        <div className="rounded-2xl bg-ink text-paper px-5 py-4 flex items-center justify-between overflow-hidden relative">
          <div>
            <div className="text-[10px] uppercase tracking-wider2 text-paper/60">
              For you
            </div>
            <div className="font-display text-xl mt-0.5 tracking-editorial">
              {currentUser?.name
                ? `Welcome, ${currentUser.name}`
                : "Personalized for you"}
            </div>
            {currentUser?.tagline && (
              <div className="text-xs text-paper/70 mt-1">{currentUser.tagline}</div>
            )}
          </div>
          <div className="hidden sm:flex items-center gap-3 text-[11px] text-paper/70">
            <button className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-paper/10 hover:bg-paper/20 transition">
              <Heart size={12} /> Saved
            </button>
            <button className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-paper/10 hover:bg-paper/20 transition">
              <ShoppingBag size={12} /> Bag (0)
            </button>
            <Sparkles size={18} className="text-gold ml-1" />
          </div>
          <div className="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-terracotta/20 blur-2xl pointer-events-none" />
        </div>
      </div>

      <AnimatePresence mode="popLayout">
        {running && items.length === 0 ? (
          <SkeletonRow key="loading" />
        ) : items.length === 0 ? (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="px-4 py-12 text-center"
          >
            <div className="surface p-8">
              <div className="font-display text-xl tracking-editorial mb-2">
                Recommendations will appear here
              </div>
              <p className="text-sm text-ash leading-relaxed max-w-sm mx-auto">
                Write your Operator Directive on the left, then press
                <span className="chip mx-1.5">Run pipeline</span>
                — MARGO will rewrite this feed in real time.
              </p>
            </div>
          </motion.div>
        ) : (
          <>
            {/* Editor's pick — Top 1 */}
            {featured && (
              <Section title="Editor's pick · this directive">
                <FeaturedCard key={featured.item_id} item={featured} />
              </Section>
            )}

            {/* More for you — Top 2..K */}
            {rest.length > 0 && (
              <Section title="More for you">
                <motion.div
                  layout
                  className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-3 gap-3"
                >
                  <AnimatePresence mode="popLayout">
                    {rest.map((it, i) => (
                      <ProductCard
                        key={it.item_id}
                        item={it}
                        rank={i + 2}
                      />
                    ))}
                  </AnimatePresence>
                </motion.div>
              </Section>
            )}
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sections
// ─────────────────────────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="px-4 py-4">
      <h3 className="text-[11px] uppercase tracking-wider2 text-ash mb-3 flex items-center justify-between">
        <span>{title}</span>
        <span className="font-mono text-[10px]">amazon · fashion</span>
      </h3>
      {children}
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Featured (hero) card
// ─────────────────────────────────────────────────────────────────────────────

function FeaturedCard({ item }: { item: CatalogItem }) {
  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.45, type: "spring", damping: 20 }}
      className="card grid grid-cols-1 md:grid-cols-2 overflow-hidden"
    >
      <div className="relative aspect-square md:aspect-auto md:min-h-[360px] bg-sand">
        {item.image_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={item.image_url}
            alt={item.title}
            className="absolute inset-0 h-full w-full object-contain p-4"
          />
        ) : (
          <Fallback id={item.item_id} />
        )}
        <div className="absolute top-3 left-3 chip-solid bg-paper text-ink shadow-sm">
          #01 · Editor's pick
        </div>
        {typeof item.score === "number" && (
          <div className="absolute top-3 right-3 chip-solid bg-ink">
            <span className="font-mono">{item.score.toFixed(2)}</span>
          </div>
        )}
      </div>
      <div className="p-5 flex flex-col">
        <div className="text-[11px] uppercase tracking-wider2 text-ash flex items-center gap-2">
          {item.brand ?? item.category?.[0] ?? "—"}
          {typeof item.rating === "number" && (
            <span className="inline-flex items-center gap-0.5 text-ink/80">
              <Star size={11} className="fill-gold text-gold" />
              {item.rating.toFixed(1)}
              {item.review_count !== undefined && (
                <span className="text-ash">· {item.review_count.toLocaleString()}</span>
              )}
            </span>
          )}
        </div>
        <h4 className="font-display text-2xl tracking-editorial leading-tight mt-1">
          {item.title}
        </h4>
        <div className="mt-2 font-mono text-lg">
          ${item.price ? Math.round(item.price) : "—"}
        </div>

        {(item.self_description || item.description) && (
          <p className="text-sm text-ink/75 leading-relaxed mt-3">
            {item.self_description ?? item.description}
          </p>
        )}

        {item.rationale && (
          <div className="mt-4 surface p-3 space-y-2">
            <div className="text-[10px] uppercase tracking-wider2 text-ash">
              Why this — three layers
            </div>
            <RationaleLine label="Personal" tone="bg-terracotta" text={item.rationale.personal} />
            <RationaleLine label="Directive" tone="bg-gold" text={item.rationale.directive} />
            <RationaleLine label="Trend" tone="bg-sage" text={item.rationale.trend} />
          </div>
        )}

        <div className="mt-auto pt-4 flex items-center gap-2">
          <button className="btn-primary">
            <ShoppingBag size={14} />
            Add to bag
          </button>
          <button className="btn-ghost">
            <Heart size={14} />
            Save
          </button>
        </div>
      </div>
    </motion.article>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid product card
// ─────────────────────────────────────────────────────────────────────────────

function ProductCard({ item, rank }: { item: CatalogItem; rank: number }) {
  const [open, setOpen] = useState(false);
  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.35, type: "spring", damping: 24 }}
      className="card group flex flex-col"
    >
      <div className="relative aspect-square bg-sand overflow-hidden">
        {item.image_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={item.image_url}
            alt={item.title}
            className="absolute inset-0 h-full w-full object-contain p-3 group-hover:scale-[1.04] transition duration-500"
          />
        ) : (
          <Fallback id={item.item_id} />
        )}
        <div className="absolute top-2 left-2 chip-solid bg-paper text-ink text-[10px]">
          #{String(rank).padStart(2, "0")}
        </div>
        {typeof item.score === "number" && (
          <div className="absolute top-2 right-2 chip-solid bg-ink text-[10px]">
            <span className="font-mono">{item.score.toFixed(2)}</span>
          </div>
        )}
        <button className="absolute bottom-2 right-2 h-8 w-8 grid place-items-center rounded-full bg-paper/90 backdrop-blur shadow-sm hover:scale-105 transition">
          <Heart size={14} />
        </button>
      </div>

      <div className="p-3 flex-1 flex flex-col">
        <div className="text-[10px] uppercase tracking-wider2 text-ash flex items-center gap-1.5">
          <span className="truncate">{item.brand ?? (item.category?.[0] ?? "—")}</span>
          {typeof item.rating === "number" && (
            <span className="inline-flex items-center gap-0.5 text-ink/80">
              <Star size={10} className="fill-gold text-gold" />
              {item.rating.toFixed(1)}
            </span>
          )}
        </div>
        <div className="font-display text-[14px] tracking-editorial leading-tight mt-0.5 line-clamp-2">
          {item.title}
        </div>
        <div className="mt-1.5 flex items-center justify-between">
          <span className="font-mono text-sm">
            ${item.price ? Math.round(item.price) : "—"}
          </span>
          <button
            onClick={() => setOpen((v) => !v)}
            className="text-[10px] uppercase tracking-wider2 text-ash hover:text-terracotta inline-flex items-center gap-1"
          >
            Why <ChevronDown
              size={11}
              className={`${open ? "rotate-180" : ""} transition`}
            />
          </button>
        </div>

        <AnimatePresence>
          {open && item.rationale && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.22 }}
              className="mt-2 border-t hairline pt-2 space-y-1.5"
            >
              {item.self_description && (
                <p className="text-[11px] italic text-ink/70 leading-relaxed">
                  “{item.self_description}”
                </p>
              )}
              <RationaleLine label="Personal" tone="bg-terracotta" text={item.rationale.personal} compact />
              <RationaleLine label="Directive" tone="bg-gold" text={item.rationale.directive} compact />
              <RationaleLine label="Trend" tone="bg-sage" text={item.rationale.trend} compact />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.article>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Bits
// ─────────────────────────────────────────────────────────────────────────────

function RationaleLine({
  label,
  text,
  tone,
  compact = false
}: {
  label: string;
  text: string;
  tone: string;
  compact?: boolean;
}) {
  return (
    <div className="flex items-start gap-2">
      <span className={`mt-1.5 h-1.5 w-1.5 rounded-full shrink-0 ${tone}`} />
      <div>
        <div className="text-[10px] uppercase tracking-wider2 text-ash">{label}</div>
        <div className={`${compact ? "text-[11px]" : "text-[12px]"} leading-relaxed text-ink/85`}>
          {text}
        </div>
      </div>
    </div>
  );
}

function Fallback({ id }: { id: string }) {
  return (
    <div className="absolute inset-0 grid place-items-center text-ash text-xs font-mono">
      {id}
    </div>
  );
}

function SkeletonRow() {
  return (
    <motion.div
      key="loading"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="p-4 space-y-4"
    >
      <div className="card overflow-hidden grid grid-cols-1 md:grid-cols-2">
        <div className="aspect-square bg-gradient-to-r from-sand via-paper to-sand bg-[length:200%_100%] animate-shimmer" />
        <div className="p-5 space-y-3">
          <div className="h-3 w-1/3 bg-sand rounded animate-shimmer bg-[length:200%_100%]" />
          <div className="h-6 w-3/4 bg-sand rounded animate-shimmer bg-[length:200%_100%]" />
          <div className="h-3 w-1/4 bg-sand rounded animate-shimmer bg-[length:200%_100%]" />
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="card overflow-hidden">
            <div className="aspect-square bg-gradient-to-r from-sand via-paper to-sand bg-[length:200%_100%] animate-shimmer" />
            <div className="p-3 space-y-2">
              <div className="h-3 w-2/3 bg-sand rounded" />
              <div className="h-3 w-1/3 bg-sand rounded" />
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
