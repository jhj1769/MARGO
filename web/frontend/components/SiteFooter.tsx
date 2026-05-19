export function SiteFooter() {
  return (
    <footer className="border-t hairline bg-paper/80 mt-24">
      <div className="container-wide py-12 grid grid-cols-1 md:grid-cols-3 gap-10">
        <div>
          <div className="font-display text-xl tracking-editorial">MARGO</div>
          <p className="text-sm text-ash mt-2 leading-relaxed max-w-sm">
            A multi-agent framework for stakeholder-aware recommendation governance.
            User · Item · Expert · Trend agents collaborating in natural language.
          </p>
        </div>

        <div className="text-sm">
          <div className="eyebrow mb-3">Project</div>
          <ul className="space-y-1.5 text-ink/80">
            <li><a className="hover:text-terracotta" href="/architecture">Framework Diagram</a></li>
            <li><a className="hover:text-terracotta" href="/demo">Interactive Demo</a></li>
            <li><a className="hover:text-terracotta" href="#paper">Paper (preprint)</a></li>
            <li><a className="hover:text-terracotta" href="#">Code · GitHub</a></li>
          </ul>
        </div>

        <div className="text-sm">
          <div className="eyebrow mb-3">About</div>
          <ul className="space-y-1.5 text-ink/80">
            <li>Research prototype</li>
            <li className="text-ash">Built for stakeholder governance research</li>
            <li className="text-ash">Open for collaboration</li>
          </ul>
        </div>
      </div>

      <div className="border-t hairline">
        <div className="container-wide py-5 flex flex-col sm:flex-row items-start sm:items-center justify-between text-xs text-ash gap-2">
          <span>© {new Date().getFullYear()} MARGO · Research prototype.</span>
          <span className="font-mono">build · margo-demo-0.1</span>
        </div>
      </div>
    </footer>
  );
}
