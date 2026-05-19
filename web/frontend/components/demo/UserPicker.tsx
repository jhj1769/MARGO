"use client";

import { useEffect, useMemo, useState } from "react";
import { Loader2, ShoppingBag, User } from "lucide-react";
import { cn } from "@/lib/cn";
import { api, type DemoUser, type HistoryItem, type UserHistory } from "@/lib/api";

/** Step 1 — pick a target user.
 *
 *  Layout: two columns (Women / Men). Clicking a user expands an inline
 *  panel beneath their row, showing their recent purchases as item cards
 *  (image · title · category breadcrumb · price). This is the User Agent's
 *  raw input rendered for the operator to inspect.
 */
export function UserPicker({
  users,
  active,
  onPick,
  currentUser
}: {
  users: DemoUser[];
  active: string;
  onPick: (id: string) => void;
  currentUser: DemoUser | null;
}) {
  const [history, setHistory] = useState<UserHistory | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch history whenever the selected user changes.
  useEffect(() => {
    if (!active) {
      setHistory(null);
      return;
    }
    let cancel = false;
    setLoading(true);
    setError(null);
    api
      .userHistory(active)
      .then((h) => {
        if (!cancel) setHistory(h);
      })
      .catch((e) => {
        if (!cancel) setError(String(e?.message ?? e));
      })
      .finally(() => {
        if (!cancel) setLoading(false);
      });
    return () => {
      cancel = true;
    };
  }, [active]);

  // Split into women / men columns. Users with no gender (mock mode) fall
  // back to a single "Shoppers" column.
  const { women, men, ungendered } = useMemo(() => {
    const w: DemoUser[] = [];
    const m: DemoUser[] = [];
    const u: DemoUser[] = [];
    for (const x of users) {
      if (x.gender === "women") w.push(x);
      else if (x.gender === "men") m.push(x);
      else u.push(x);
    }
    return { women: w, men: m, ungendered: u };
  }, [users]);

  if (users.length === 0) return null;

  return (
    <div className="surface flex flex-col">
      {/* header */}
      <div className="px-4 py-3 border-b hairline flex items-center justify-between">
        <div>
          <div className="eyebrow flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-terracotta" />
            Step 1 · Pick a target user
          </div>
          <div className="font-display tracking-editorial text-lg">
            Whose feed should we re-rank?
          </div>
        </div>
        {currentUser && (
          <span className="chip">
            <User size={12} className="text-terracotta" />
            {currentUser.name ?? currentUser.user_id}
          </span>
        )}
      </div>

      {/* two-column user grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4 p-4">
        {ungendered.length > 0 ? (
          <div className="md:col-span-2">
            <ColumnHeader label="Shoppers" count={ungendered.length} />
            <UserGrid
              users={ungendered}
              active={active}
              onPick={onPick}
            />
          </div>
        ) : (
          <>
            <div>
              <ColumnHeader label="Women" count={women.length} />
              <UserGrid users={women} active={active} onPick={onPick} />
            </div>
            <div>
              <ColumnHeader label="Men" count={men.length} />
              <UserGrid users={men} active={active} onPick={onPick} />
            </div>
          </>
        )}
      </div>

      {/* accordion: history of the selected user */}
      {active && (
        <div className="border-t hairline bg-paper/60 px-4 py-4">
          <div className="flex items-baseline justify-between mb-3">
            <div>
              <div className="eyebrow">Interaction history</div>
              <div className="text-sm text-ink/80">
                {currentUser?.name ?? active}
                {history?.history_size != null && (
                  <span className="text-ash">
                    {" "}
                    · {history.history_size} interactions on record
                    {history.items.length < (history.history_size ?? 0) && (
                      <> · showing {history.items.length} most recent</>
                    )}
                  </span>
                )}
              </div>
            </div>
            {loading && (
              <span className="chip text-ash">
                <Loader2 size={12} className="animate-spin" /> loading…
              </span>
            )}
          </div>

          {error && (
            <div className="text-xs text-terracotta">{error}</div>
          )}

          {history?.summary && (
            <p className="text-xs text-ink/65 leading-relaxed mb-3 italic">
              “{history.summary}”
            </p>
          )}

          {history && history.items.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
              {history.items.map((it) => (
                <HistoryItemCard key={it.item_id} item={it} />
              ))}
            </div>
          )}

          {history && history.items.length === 0 && !loading && (
            <div className="text-xs text-ash flex items-center gap-2">
              <ShoppingBag size={14} />
              No item-level history available for this user
              (mock mode shows summaries only).
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ColumnHeader({ label, count }: { label: string; count: number }) {
  return (
    <div className="flex items-baseline justify-between mb-2">
      <div className="eyebrow">{label}</div>
      <span className="text-[10px] font-mono text-ash">{count}</span>
    </div>
  );
}

function UserGrid({
  users,
  active,
  onPick
}: {
  users: DemoUser[];
  active: string;
  onPick: (id: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {users.map((u) => {
        const isActive = u.user_id === active;
        return (
          <button
            key={u.user_id}
            onClick={() => onPick(u.user_id)}
            className={cn(
              "rounded-full px-3 py-1.5 text-xs font-medium border transition",
              isActive
                ? "bg-ink text-paper border-ink"
                : "border-ink/10 hover:border-ink/30 text-ink"
            )}
            title={u.summary ?? u.user_id}
          >
            <User size={11} className="inline -mt-0.5 mr-1.5" />
            {u.name ?? u.user_id}
            {u.history_size != null && (
              <span className={cn(
                "ml-1.5 font-mono text-[10px]",
                isActive ? "text-paper/70" : "text-ash"
              )}>
                {u.history_size}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

/** A single item from a user's interaction history. */
function HistoryItemCard({ item }: { item: HistoryItem }) {
  // categories[0] is usually the marketplace root ("Clothing, Shoes & Jewelry");
  // [1] is gender; the meaningful breadcrumb starts at [2].
  const crumb = (item.category ?? [])
    .slice(2)
    .filter(Boolean)
    .slice(0, 3)
    .join(" › ");
  const price =
    typeof item.price === "number" && !Number.isNaN(item.price)
      ? `$${item.price.toFixed(2)}`
      : "—";
  return (
    <div className="rounded-lg border hairline bg-paper overflow-hidden flex flex-col">
      <div className="aspect-square bg-sand/50 flex items-center justify-center overflow-hidden">
        {item.image_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={item.image_url}
            alt={item.title}
            className="h-full w-full object-contain p-2"
            loading="lazy"
          />
        ) : (
          <ShoppingBag size={20} className="text-ash" />
        )}
      </div>
      <div className="p-2 flex-1 flex flex-col gap-1">
        <div
          className="text-[11px] leading-snug text-ink line-clamp-2"
          title={item.title}
        >
          {item.title || "(no title)"}
        </div>
        {crumb && (
          <div className="text-[10px] text-ash truncate" title={crumb}>
            {crumb}
          </div>
        )}
        <div className="mt-auto flex items-baseline justify-between">
          <span className="font-mono text-[10px] text-terracotta">{price}</span>
          {item.brand && (
            <span className="text-[10px] text-ash truncate ml-2" title={item.brand}>
              {item.brand}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
