import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canopy: {
          950: "#0a0f0a",
          900: "#111a11",
          800: "#1b2a1b",
          700: "#2a3f28",
        },
        moss: {
          600: "#4a6741",
          500: "#5e7d52",
          400: "#7c9a6e",
        },
        sage: {
          200: "#d9e2cf",
          100: "#e9eee2",
        },
        glow: {
          400: "#e8dcae",
          300: "#f1e9c9",
        },
        paper: "#f8f6ef",
        ink: "#1b2017",
      },
      fontFamily: {
        display: ["var(--font-fraunces)", "serif"],
        body: ["var(--font-manrope)", "sans-serif"],
      },
      keyframes: {
        drift: {
          "0%, 100%": { transform: "translate(0, 0)" },
          "50%": { transform: "translate(8px, -14px)" },
        },
        flicker: {
          "0%, 100%": { opacity: "0.55" },
          "50%": { opacity: "0.85" },
        },
      },
      animation: {
        drift: "drift 7s ease-in-out infinite",
        flicker: "flicker 4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
export default config;
