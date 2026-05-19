// Pure-client mock fallback so the marketing pages render even without the
// FastAPI backend running. Demo interactions still require the backend.

import type { CatalogItem, Scenario } from "./api";

export const FALLBACK_SCENARIOS: Scenario[] = [
  {
    id: "casual_to_formal",
    title: "Campaign · Casual → Formal Upsell",
    blurb: "Migrate casual shoppers toward their first structured piece without breaking trust.",
    default_user: "u_minji",
    directive: {
      goal: "Casual → formal upsell",
      natural_language:
        "Move casual shoppers toward their first formal capsule piece. Keep the price jump comfortable. Boost trench coats and tailored outerwear.",
      structured_constraints: {
        price_diff_pct_max: 30,
        boost_category: ["trench-coat", "blazer"]
      }
    }
  }
];

export const FALLBACK_ITEMS: CatalogItem[] = [
  {
    item_id: "demo_a",
    title: "Beige Midi Trench Coat",
    price: 168,
    category: ["outerwear", "trench-coat"],
    color: "beige",
    material: "cotton-blend",
    image_url:
      "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=800&q=80",
    brand: "Maison Ardent"
  }
];
