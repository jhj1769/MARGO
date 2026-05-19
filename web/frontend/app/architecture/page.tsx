import { Section } from "@/components/Section";
import { PhaseDiagram } from "@/components/PhaseDiagram";

export default function ArchitecturePage() {
  return (
    <>
      <section className="border-b hairline">
        <div className="container-wide pt-16 pb-10">
          <div className="eyebrow">Framework</div>
          <h1 className="font-display tracking-editorial text-[clamp(2rem,4.5vw,3.8rem)] leading-[1.05] mt-3 max-w-3xl">
            Four agents, four phases — and runtime guardrails on every message.
          </h1>
          <p className="text-ink/75 mt-6 max-w-2xl leading-relaxed">
            MARGO turns operator intent and live external context into first-class
            participants in the recommendation pipeline. Each phase is bounded by a
            typed message protocol, so natural-language reasoning never leaves the
            framework as untyped text.
          </p>
        </div>
      </section>

      <Section>
        <PhaseDiagram />
      </Section>

      {/* ───────── Lifecycle detail ───────── */}
      <section className="border-t hairline bg-sand/30">
        <div className="container-wide py-20 grid grid-cols-1 lg:grid-cols-2 gap-x-12 gap-y-10">
          {PHASES.map((p) => (
            <article key={p.id} className="">
              <div className="flex items-baseline gap-3">
                <div
                  className="font-display text-5xl tracking-editorial text-terracotta"
                  style={{ lineHeight: 1 }}
                >
                  {p.id}
                </div>
                <h3 className="font-display text-2xl tracking-editorial">
                  {p.title}
                </h3>
              </div>
              <p className="text-ink/80 leading-relaxed mt-3">{p.blurb}</p>
              <ul className="mt-5 space-y-2 text-sm">
                {p.bullets.map((b) => (
                  <li
                    key={b}
                    className="flex items-start gap-3 before:content-[''] before:w-1.5 before:h-1.5 before:bg-ink/30 before:rounded-full before:mt-2 before:shrink-0"
                  >
                    {b}
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </section>

    </>
  );
}

const PHASES = [
  {
    id: "01",
    title: "Agent Initialization",
    blurb:
      "Offline pre-computation. User histories are turned into NL personas; item metadata becomes structured + textual self-knowledge. The phase runs once per dataset and is cached on disk.",
    bullets: [
      "User Agent .build_profile() — synthesises an NL persona from the user's history.",
      "Item Agent — backed by ItemFacts (immutable attributes; no fabrication).",
      "Snapshot store ready for the Trend Agent (per-(domain, time_window) cache)."
    ]
  },
  {
    id: "02",
    title: "Directive Generation",
    blurb:
      "Online. The Expert Agent issues a hybrid directive (structured constraints + NL). The Trend Agent interprets web evidence and broadcasts a TrendInterpretation to every other agent.",
    bullets: [
      "Hard constraints (price gap, forbidden categories) are mechanically checkable.",
      "Soft NL captures intent that no rule can fully express.",
      "Trend keywords feed directly into the directive-aware retrieval query."
    ]
  },
  {
    id: "03",
    title: "Multi-Agent Reasoning",
    blurb:
      "BGE-M3 fetches a directive-aware candidate pool. Each Item Agent rewrites itself in context. The User Agent ranks and emits a Top-K with a 3-layer rationale.",
    bullets: [
      "Candidate pool changes when the directive changes — not just the ranking.",
      "Item Agents anchor every claim to a real attribute (anti-hallucination).",
      "Rationale is mandatory, structured, and three-layered."
    ]
  },
  {
    id: "04",
    title: "Validation & Refinement",
    blurb:
      "The Expert Agent re-checks the Top-K. Structured violations are caught mechanically; soft NL compliance is scored. If the score falls below threshold, the directive is refined and Phase 2 re-runs.",
    bullets: [
      "max_iterations = 3 to prevent runaway loops.",
      "Convergence at compliance_score ≥ 0.85.",
      "Every refinement is recorded toward the CADR governance metric."
    ]
  }
];
