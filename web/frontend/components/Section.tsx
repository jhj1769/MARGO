import { cn } from "@/lib/cn";
import type { ReactNode } from "react";

export function Section({
  eyebrow,
  title,
  subtitle,
  children,
  className,
  align = "left"
}: {
  eyebrow?: string;
  title?: string;
  subtitle?: string;
  children?: ReactNode;
  className?: string;
  align?: "left" | "center";
}) {
  return (
    <section className={cn("py-20", className)}>
      <div className={cn("container-wide", align === "center" && "text-center")}>
        {(eyebrow || title || subtitle) && (
          <header className={cn("mb-10 max-w-3xl", align === "center" && "mx-auto")}>
            {eyebrow && <div className="eyebrow mb-3">{eyebrow}</div>}
            {title && (
              <h2 className="font-display tracking-editorial text-3xl sm:text-4xl text-ink">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-4 text-ash leading-relaxed">{subtitle}</p>
            )}
          </header>
        )}
        {children}
      </div>
    </section>
  );
}
