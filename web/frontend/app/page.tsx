import Link from "next/link";
import { ArrowRight, ArrowUpRight, BadgeCheck, Layers, ShieldCheck } from "lucide-react";
import { Section } from "@/components/Section";
import { AgentRing } from "@/components/AgentRing";
import { HeroVisual } from "@/components/HeroVisual";
import { RationalePreview } from "@/components/RationalePreview";

const AGENTS = [
  {
    id: "user",
    label: "User Agent",
    role: "Represents the shopper",
    skills: ["query_preference", "evaluate_candidate", "update_profile"],
    blurb:
      "Speaks for one shopper in natural language. Weighs personal taste against the operator's intent and the trend signal — openly, not invisibly."
  },
  {
    id: "item",
    label: "Item Agent",
    role: "Self-describes under context",
    skills: ["self_describe", "update_reflection"],
    blurb:
      "Each item rewrites itself under the current directive and trend, anchored only to its true attributes — never inventing what it isn't."
  },
  {
    id: "expert",
    label: "Expert Agent",
    role: "Governs through directives",
    skills: ["issue_directive", "validate_recommendation", "refine_directive"],
    blurb:
      "Translates operator intent into machine-checkable + natural-language hybrid directives. Validates the Top-K and refines until compliance holds."
  },
  {
    id: "trend",
    label: "Trend Agent",
    role: "Interprets external context",
    skills: ["query_trend", "interpret_trend", "broadcast"],
    blurb:
      "Reads market evidence and rewrites it for the recommendation context, then broadcasts the interpretation to every other agent."
  }
];

const DEMO_CARDS = [
  {
    href: "/demo",
    title: "Live recommendation governance",
    venue: "Interactive demo",
    image:
      "https://images.unsplash.com/photo-1485518882345-15568b007407?w=1600&q=80",
    blurb:
      "Type an operator brief in plain English and watch one shopper's feed reorganise itself — pool, ranking, and rationale all updating together."
  },
  {
    href: "/architecture",
    title: "4-Phase lifecycle diagram",
    venue: "Framework",
    image:
      "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=1600&q=80",
    blurb:
      "Initialization → Directive Generation → Multi-Agent Reasoning → Validation & Refinement. Hover any node to see the messages it accepts and emits."
  },
  {
    href: "#paper",
    title: "Paper · stakeholder-aware governance",
    venue: "Preprint",
    image:
      "https://images.unsplash.com/photo-1455390582262-044cdead277a?w=1600&q=80",
    blurb:
      "How four LLM agents and a runtime guardrail layer close the governance gap between collaborative-filtering models and operational reality."
  }
];

export default function HomePage() {
  return (
    <>
      {/* ───────────────────────── Hero ───────────────────────── */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid opacity-60 pointer-events-none" />
        <div className="container-wide pt-20 pb-24 relative grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          <div className="lg:col-span-7 max-w-2xl">
            <div className="inline-flex items-center gap-2 chip">
              <span className="h-1.5 w-1.5 rounded-full bg-terracotta" />
              Research prototype · 2026
            </div>

            <h1 className="font-display tracking-editorial mt-6 text-[clamp(2.2rem,4.6vw,3.8rem)] leading-[1.08] text-ink">
              Recommendations that{" "}
              <em className="font-normal text-terracotta not-italic">follow your direction.</em>
            </h1>

            <p className="mt-6 text-lg text-ink/75 leading-relaxed max-w-xl">
              Real merchandising is shaped by more than clicks — campaigns,
              seasonal pushes, pricing rules, and what the market is doing this
              week. <span className="font-display italic text-ink">MARGO</span>{" "}
              lets your team write that direction in plain English and turns it
              into a directive the recommender actually follows. Every Top-K
              item explains itself in three layers: what fits the shopper,
              what serves your direction, and how it tracks the trend.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link href="/demo" className="btn-primary group">
                Try the demo
                <ArrowRight size={16} className="transition group-hover:translate-x-0.5" />
              </Link>
              <Link href="/architecture" className="btn-ghost">
                How it works
              </Link>
              <a href="#paper" className="btn-ghost">
                Paper preprint
                <ArrowUpRight size={14} />
              </a>
            </div>
          </div>

          <div className="lg:col-span-5 lg:sticky lg:top-24">
            <HeroVisual />
          </div>
        </div>
      </section>

      {/* ───────────────────────── Abstract ───────────────────────── */}
      <section className="border-t hairline bg-sand/30">
        <div className="container-wide py-20 grid grid-cols-1 lg:grid-cols-12 gap-12">
          <div className="lg:col-span-3">
            <div className="eyebrow">Abstract</div>
            <div className="font-display text-3xl tracking-editorial mt-3 leading-tight">
              The governance gap.
            </div>
          </div>
          <div className="lg:col-span-9 text-ink/85 leading-relaxed text-[1.05rem]">
            <p>
              Industrial recommendation is shaped by far more than user–item history.
              Marketing intent, business rules, regulatory constraints and live market
              context all bend ranking in production — yet none of them live inside
              the interaction logs that learning models consume.
            </p>
            <p className="mt-4">
              MARGO rephrases the problem as one of <em>governance</em>. A four-agent
              collaboration — User, Item, Expert, Trend — operates entirely in natural
              language, but is bounded by a typed message protocol and a set of
              runtime guardrails that audit each message against the catalogue,
              the domain vocabulary, the schema, and cross-agent agreement.
            </p>
          </div>
        </div>
      </section>

      {/* ───────────────────────── 4 Agents ───────────────────────── */}
      <Section
        eyebrow="Stakeholders"
        title="Four agents · one decision"
        subtitle="Each agent owns a narrow surface of behaviour. Together they replace the implicit, hidden compromises of production ranking with explicit, auditable negotiations."
      >
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
          <div className="lg:col-span-5">
            <AgentRing />
          </div>
          <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-4">
            {AGENTS.map((a) => (
              <article
                key={a.id}
                className="card p-5 hover:border-ink/20 transition group"
              >
                <div className="flex items-baseline justify-between">
                  <h3 className="font-display text-xl">{a.label}</h3>
                  <span className="text-[10px] uppercase tracking-wider2 text-ash">
                    {a.id}
                  </span>
                </div>
                <p className="text-xs text-ash mt-1">{a.role}</p>
                <p className="text-sm mt-3 text-ink/85 leading-relaxed">{a.blurb}</p>
                <div className="flex flex-wrap gap-1.5 mt-4">
                  {a.skills.map((s) => (
                    <span key={s} className="chip font-mono text-[10px]">
                      {s}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </div>
      </Section>

      {/* ───────────────────────── 3-Layer Rationale Preview ───────────────────────── */}
      <section className="border-t hairline bg-ink text-paper">
        <div className="container-wide py-20 grid grid-cols-1 lg:grid-cols-12 gap-10">
          <div className="lg:col-span-5">
            <div className="eyebrow text-paper/60">Output</div>
            <h2 className="font-display tracking-editorial text-4xl mt-3 leading-tight">
              Every recommendation, three reasons.
            </h2>
            <p className="text-paper/70 leading-relaxed mt-4 max-w-md">
              MARGO never hides why an item surfaces. Each Top-K position carries a
              three-layer rationale — what fits <em>you</em>, what serves the
              <em> operator's</em> intent, and how it aligns with <em>the world</em>.
            </p>

            <ul className="mt-8 space-y-3 text-sm text-paper/80">
              <li className="flex items-start gap-3">
                <BadgeCheck size={18} className="text-gold mt-0.5 shrink-0" />
                Hard structured constraints (price gap, forbidden categories) are
                checked mechanically — the LLM cannot lie its way past them.
              </li>
              <li className="flex items-start gap-3">
                <ShieldCheck size={18} className="text-sage mt-0.5 shrink-0" />
                Soft NL intent is graded by the Expert Agent and the Top-K is
                refined when compliance drops below threshold.
              </li>
              <li className="flex items-start gap-3">
                <Layers size={18} className="text-terracotta mt-0.5 shrink-0" />
                Trend context is broadcast, not assumed — every interpretation is
                cached for full reproducibility.
              </li>
            </ul>
          </div>
          <div className="lg:col-span-7">
            <RationalePreview />
          </div>
        </div>
      </section>

      {/* ───────────────────────── Demo Cards ───────────────────────── */}
      <Section eyebrow="Explore" title="See MARGO in motion">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {DEMO_CARDS.map((d, i) => (
            <Link
              key={d.title}
              href={d.href}
              className="card group hover:-translate-y-0.5 hover:shadow-xl hover:shadow-ink/5 transition"
            >
              <div className="relative aspect-[4/3] overflow-hidden bg-sand">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={d.image}
                  alt=""
                  className="h-full w-full object-cover saturate-[.85] group-hover:scale-[1.03] transition duration-500"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-ink/40 via-transparent" />
                <div className="absolute top-3 left-3 chip-solid">{d.venue}</div>
              </div>
              <div className="p-5">
                <h3 className="font-display text-xl tracking-editorial flex items-start gap-2">
                  {d.title}
                  <ArrowUpRight
                    size={16}
                    className="mt-1 -translate-y-px text-ash group-hover:text-terracotta group-hover:translate-x-0.5 transition"
                  />
                </h3>
                <p className="text-sm text-ink/70 mt-2 leading-relaxed">{d.blurb}</p>
              </div>
              <div className="px-5 py-3 border-t hairline flex items-center justify-between text-xs text-ash">
                <span>{i === 0 ? "Live demo" : i === 1 ? "Interactive" : "PDF · arXiv"}</span>
                <span className="font-mono">{`demo / 0${i + 1}`}</span>
              </div>
            </Link>
          ))}
        </div>
      </Section>

      {/* ───────────────────────── About this prototype ───────────────────────── */}
      <section id="paper" className="border-t hairline">
        <div className="container-wide py-20 grid grid-cols-1 lg:grid-cols-12 gap-12">
          <div className="lg:col-span-4">
            <div className="eyebrow">About</div>
            <h2 className="font-display tracking-editorial text-3xl mt-3 leading-tight">
              A research prototype, built to be reused.
            </h2>
          </div>

          <div className="lg:col-span-8 text-ink/80 leading-relaxed text-[1.02rem] space-y-4">
            <p>
              MARGO is an exploration into bridging the gap between LLM-based
              recommendation and the governance reality of real businesses. Operator
              briefs, business rules, and live market trends are first-class
              participants — not silent filters bolted on top of a ranking model.
            </p>
            <p>
              The codebase is structured as a library so the same engine can drive
              evaluation runs, this interactive demo, or be embedded into downstream
              applications. Bring your own catalogue, swap the LLM, plug in a
              different trend source — every component is replaceable.
            </p>
          </div>
        </div>
      </section>
    </>
  );
}

