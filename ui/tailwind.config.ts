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
        void: "rgb(var(--void) / <alpha-value>)",
        surface: {
          DEFAULT: "rgb(var(--surface) / <alpha-value>)",
          hover: "rgb(var(--surface-hover) / <alpha-value>)",
          inset: "rgb(var(--surface-inset) / <alpha-value>)",
        },
        border: {
          DEFAULT: "rgb(var(--border) / <alpha-value>)",
          light: "rgb(var(--border-light) / <alpha-value>)",
        },
        text: {
          DEFAULT: "rgb(var(--text) / <alpha-value>)",
          secondary: "rgb(var(--text-secondary) / <alpha-value>)",
          muted: "rgb(var(--text-muted) / <alpha-value>)",
        },
        amber: {
          DEFAULT: "rgb(var(--amber) / <alpha-value>)",
          hover: "rgb(var(--amber-hover) / <alpha-value>)",
          soft: "rgb(var(--amber-soft) / <alpha-value>)",
          softer: "rgb(var(--amber-softer) / <alpha-value>)",
        },
        cyan: {
          DEFAULT: "rgb(var(--cyan) / <alpha-value>)",
          soft: "rgb(var(--cyan-soft) / <alpha-value>)",
        },
        violet: {
          DEFAULT: "rgb(var(--violet) / <alpha-value>)",
          soft: "rgb(var(--violet-soft) / <alpha-value>)",
        },
        emerald: {
          DEFAULT: "rgb(var(--emerald) / <alpha-value>)",
          soft: "rgb(var(--emerald-soft) / <alpha-value>)",
        },
        rose: {
          DEFAULT: "rgb(var(--rose) / <alpha-value>)",
          soft: "rgb(var(--rose-soft) / <alpha-value>)",
        },
        gold: {
          DEFAULT: "rgb(var(--gold) / <alpha-value>)",
          soft: "rgb(var(--gold-soft) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: ["Outfit", "system-ui", "sans-serif"],
        display: ["Fraunces", "Georgia", "serif"],
        mono: ["Space Mono", "Consolas", "monospace"],
      },
      keyframes: {
        "breathe": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
        "rise": {
          "0%": { transform: "translateY(16px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        "slide-in": {
          "0%": { transform: "translateX(-20px)", opacity: "0" },
          "100%": { transform: "translateX(0)", opacity: "1" },
        },
        "scale-up": {
          "0%": { transform: "scale(0.92)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
        "glow": {
          "0%, 100%": { boxShadow: "0 0 8px rgb(var(--amber) / 0.2)" },
          "50%": { boxShadow: "0 0 24px rgb(var(--amber) / 0.4)" },
        },
        "flow": {
          "0%": { opacity: "0.3" },
          "50%": { opacity: "1" },
          "100%": { opacity: "0.3" },
        },
        "shimmer": {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        "breathe": "breathe 2s ease-in-out infinite",
        "rise": "rise 0.5s cubic-bezier(0.22, 1, 0.36, 1)",
        "slide-in": "slide-in 0.35s cubic-bezier(0.22, 1, 0.36, 1)",
        "scale-up": "scale-up 0.3s cubic-bezier(0.22, 1, 0.36, 1)",
        "glow": "glow 3s ease-in-out infinite",
        "flow": "flow 2s ease-in-out infinite",
        "shimmer": "shimmer 2s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
