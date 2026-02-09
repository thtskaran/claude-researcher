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
        // Status colors
        success: "#10b981",
        warning: "#f59e0b",
        error: "#ef4444",
        info: "#3b82f6",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
  darkMode: "class",
};

export default config;
