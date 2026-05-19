import { DemoWorkbench } from "@/components/demo/DemoWorkbench";
import { MousePointerClick, PenLine, Play } from "lucide-react";

export const dynamic = "force-dynamic";

const STEPS = [
  {
    icon: MousePointerClick,
    label: "Pick a target user",
    detail: "Choose one user — Women or Men column — and inspect their interaction history."
  },
  {
    icon: PenLine,
    label: "Issue an Operator Directive",
    detail: "Goal + plain-English directive + optional structured constraints (machine-checkable)."
  },
  {
    icon: Play,
    label: "Run the pipeline",
    detail: "Watch the Top-K ranked list re-form and the 4-phase agent messages stream in real time."
  }
];

export default function DemoPage() {
  return (
    <div className="border-t hairline">
      <div className="container-wide py-6">
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
          <div>
            <div className="eyebrow">Interactive demo</div>
            <h1 className="font-display text-3xl tracking-editorial mt-2">
              Watch governance happen.
            </h1>
            <p className="text-sm text-ash mt-1 max-w-xl">
              Pick a target user, issue an Operator Directive, and see the Top-K
              ranked list re-form — candidate pool, ranking, and rationale all
              updating together.
            </p>
          </div>
          <a
            href="/architecture"
            className="hidden md:inline-flex btn-ghost text-sm shrink-0"
          >
            Read the framework →
          </a>
        </div>

        {/* 3-step onboarding hint */}
        <ol className="mt-5 grid grid-cols-1 sm:grid-cols-3 gap-3">
          {STEPS.map((s, i) => {
            const Icon = s.icon;
            return (
              <li
                key={s.label}
                className="surface px-3 py-3 flex items-start gap-3"
              >
                <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-ink text-paper text-xs font-mono">
                  {i + 1}
                </span>
                <div className="min-w-0">
                  <div className="flex items-center gap-1.5 text-sm font-medium">
                    <Icon size={14} className="text-terracotta shrink-0" />
                    {s.label}
                  </div>
                  <p className="text-xs text-ash mt-0.5 leading-relaxed">
                    {s.detail}
                  </p>
                </div>
              </li>
            );
          })}
        </ol>
      </div>

      <DemoWorkbench />
    </div>
  );
}
