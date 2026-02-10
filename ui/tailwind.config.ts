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
        // Primary brand color from design system
        primary: {
          DEFAULT: "#2b7cee",
          light: "#5ca3ff",
          dark: "#1a5fc9",
        },
        // Dark theme colors
        dark: {
          bg: "#101822",
          surface: "#1a2332",
          border: "#2a3544",
        },
        // Surface variants from mockups
        "surface-highlight": "#252a33",
        "card-dark": "#161e2a",
        "card-hover": "#1c2635",
        // Terminal / logs palette
        "terminal-black": "#0d1117",
        "terminal-border": "#30363d",
        // Accent colors
        "accent-success": "#10b981",
        "accent-blue": "#58a6ff",
        "accent-green": "#3fb950",
        "accent-yellow": "#d29922",
        "accent-red": "#f85149",
        "accent-neutral": "#64748b",
        // Status colors
        success: "#10b981",
        warning: "#f59e0b",
        error: "#ef4444",
        info: "#3b82f6",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      keyframes: {
        "pulse-glow": {
          "0%": { transform: "scale(1)", opacity: "0.9" },
          "50%": { transform: "scale(1.05)", opacity: "1" },
          "100%": { transform: "scale(1)", opacity: "0.9" },
        },
        "flow-pulse": {
          "0%": { opacity: "0.3" },
          "50%": { opacity: "1" },
          "100%": { opacity: "0.3" },
        },
      },
      animation: {
        "pulse-glow": "pulse-glow 3s infinite ease-in-out",
        "flow-pulse": "flow-pulse 2s infinite ease-in-out",
      },
    },
  },
  plugins: [],
  darkMode: "class",
};

export default config;
