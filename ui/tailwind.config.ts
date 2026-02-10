import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Theme-adaptive neutrals (CSS variable RGB triplets)
        page: "rgb(var(--page) / <alpha-value>)",
        card: {
          DEFAULT: "rgb(var(--card) / <alpha-value>)",
          hover: "rgb(var(--card-hover) / <alpha-value>)",
          inset: "rgb(var(--card-inset) / <alpha-value>)",
        },
        edge: {
          DEFAULT: "rgb(var(--edge) / <alpha-value>)",
          light: "rgb(var(--edge-light) / <alpha-value>)",
        },
        ink: {
          DEFAULT: "rgb(var(--ink) / <alpha-value>)",
          secondary: "rgb(var(--ink-secondary) / <alpha-value>)",
          muted: "rgb(var(--ink-muted) / <alpha-value>)",
        },
        // Primary: deep sage green
        sage: {
          DEFAULT: "rgb(var(--sage) / <alpha-value>)",
          hover: "rgb(var(--sage-hover) / <alpha-value>)",
          soft: "rgb(var(--sage-soft) / <alpha-value>)",
          softer: "rgb(var(--sage-softer) / <alpha-value>)",
        },
        // Secondary: warm terracotta
        terra: {
          DEFAULT: "rgb(var(--terra) / <alpha-value>)",
          soft: "rgb(var(--terra-soft) / <alpha-value>)",
        },
        // Tertiary: dusty iris purple
        iris: {
          DEFAULT: "rgb(var(--iris) / <alpha-value>)",
          soft: "rgb(var(--iris-soft) / <alpha-value>)",
        },
        // Success: olive green
        olive: {
          DEFAULT: "rgb(var(--olive) / <alpha-value>)",
          soft: "rgb(var(--olive-soft) / <alpha-value>)",
        },
        // Warning: golden
        gold: {
          DEFAULT: "rgb(var(--gold) / <alpha-value>)",
          soft: "rgb(var(--gold-soft) / <alpha-value>)",
        },
        // Error: warm coral
        coral: {
          DEFAULT: "rgb(var(--coral) / <alpha-value>)",
          soft: "rgb(var(--coral-soft) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: ["Figtree", "system-ui", "sans-serif"],
        display: ["Newsreader", "Georgia", "serif"],
        mono: ["IBM Plex Mono", "monospace"],
      },
      keyframes: {
        "soft-pulse": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
        "fade-up-in": {
          "0%": { transform: "translateY(12px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        "slide-in-left": {
          "0%": { transform: "translateX(-16px)", opacity: "0" },
          "100%": { transform: "translateX(0)", opacity: "1" },
        },
        "scale-in": {
          "0%": { transform: "scale(0.95)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
        "glow-pulse": {
          "0%, 100%": { boxShadow: "0 0 8px rgb(var(--sage) / 0.3)" },
          "50%": { boxShadow: "0 0 20px rgb(var(--sage) / 0.5)" },
        },
        "flow-pulse": {
          "0%": { opacity: "0.3" },
          "50%": { opacity: "1" },
          "100%": { opacity: "0.3" },
        },
      },
      animation: {
        "soft-pulse": "soft-pulse 2s ease-in-out infinite",
        "fade-up-in": "fade-up-in 0.4s ease-out",
        "slide-in-left": "slide-in-left 0.3s ease-out",
        "scale-in": "scale-in 0.25s ease-out",
        "glow-pulse": "glow-pulse 3s ease-in-out infinite",
        "flow-pulse": "flow-pulse 2s infinite ease-in-out",
      },
    },
  },
  plugins: [],
};

export default config;
