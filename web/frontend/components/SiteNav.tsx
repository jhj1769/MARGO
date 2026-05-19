"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/cn";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/architecture", label: "Architecture" },
  { href: "/demo", label: "Demo" }
];

export function SiteNav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b hairline bg-paper/85 backdrop-blur">
      <div className="container-wide flex h-14 items-center justify-between">
        <Link
          href="/"
          className="flex items-center gap-2 font-display text-lg tracking-editorial"
        >
          <span className="font-bold">MARGO</span>
          <span className="hidden sm:inline text-ash text-xs font-sans font-normal">
            A Multi-Agent Framework for Stakeholder-Aware Recommendation Governance
          </span>
        </Link>

        <nav className="flex items-center gap-1">
          {NAV.map((n) => {
            const active =
              pathname === n.href ||
              (n.href !== "/" && pathname?.startsWith(n.href));
            return (
              <Link
                key={n.href}
                href={n.href}
                className={cn(
                  "px-3 py-1.5 rounded-full text-sm font-medium transition",
                  active
                    ? "bg-ink text-paper"
                    : "text-ink/70 hover:text-ink hover:bg-ink/5"
                )}
              >
                {n.label}
              </Link>
            );
          })}
          <a
            href="https://github.com/jhj1769/MARGO"
            target="_blank"
            rel="noreferrer"
            className="ml-2 hidden sm:inline-flex btn-ghost h-8 px-3 text-xs"
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
