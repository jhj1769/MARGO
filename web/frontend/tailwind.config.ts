import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx,mdx}",
    "./components/**/*.{ts,tsx,mdx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        // editorial fashion palette
        ink:      "#0F1B2D",   // deep navy ink
        paper:    "#FAF8F4",   // off-white background
        sand:     "#EDE7DA",   // muted card surface
        ash:      "#7B8597",
        terracotta: "#C4664E",
        sage:     "#7F8B5A",
        gold:     "#B68F4F"
      },
      fontFamily: {
        display: ["var(--font-display)", "Fraunces", "ui-serif", "Georgia", "serif"],
        sans:    ["var(--font-sans)", "Inter", "ui-sans-serif", "system-ui"],
        mono:    ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"]
      },
      letterSpacing: {
        editorial: "-0.02em",
        wider2:    "0.18em"
      },
      keyframes: {
        flow: {
          "0%":   { strokeDashoffset: "24" },
          "100%": { strokeDashoffset: "0" }
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" }
        },
        rise: {
          "0%":   { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        }
      },
      animation: {
        flow:    "flow 2s linear infinite",
        shimmer: "shimmer 2.4s linear infinite",
        rise:    "rise .45s ease-out"
      }
    }
  },
  plugins: []
};
export default config;
