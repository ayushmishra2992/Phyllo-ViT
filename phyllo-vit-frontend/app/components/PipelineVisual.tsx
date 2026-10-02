"use client";

const STEPS = [
  { key: "image", label: "Leaf image" },
  { key: "vit", label: "ViT encoder" },
  { key: "class", label: "Phyllotaxy" },
  { key: "embed", label: "768-D embedding" },
  { key: "aqi", label: "AQI regression" },
] as const;

interface PipelineVisualProps {
  /** index of the currently active step, -1 = idle, STEPS.length = all complete */
  activeStep: number;
}

export default function PipelineVisual({ activeStep }: PipelineVisualProps) {
  return (
    <div className="flex w-full items-center justify-between gap-1 sm:gap-2">
      {STEPS.map((step, i) => {
        const isDone = activeStep > i || activeStep >= STEPS.length;
        const isActive = activeStep === i;
        return (
          <div key={step.key} className="flex flex-1 flex-col items-center">
            <div className="flex w-full items-center">
              {i > 0 && (
                <div
                  className={`h-px flex-1 transition-colors duration-500 ${
                    isDone || isActive ? "bg-moss-500" : "bg-sage-200"
                  }`}
                />
              )}
              <div
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-[10px] font-semibold transition-all duration-300 sm:h-8 sm:w-8 ${
                  isDone
                    ? "border-moss-500 bg-moss-500 text-white"
                    : isActive
                      ? "border-moss-500 bg-glow-300 text-canopy-900 animate-flicker"
                      : "border-sage-200 bg-white text-moss-400"
                }`}
              >
                {isDone ? "✓" : i + 1}
              </div>
              {i < STEPS.length - 1 && (
                <div
                  className={`h-px flex-1 transition-colors duration-500 ${
                    isDone ? "bg-moss-500" : "bg-sage-200"
                  }`}
                />
              )}
            </div>
            <span
              className={`mt-2 text-center text-[10px] leading-tight sm:text-xs ${
                isDone || isActive
                  ? "font-medium text-canopy-800"
                  : "text-moss-400"
              }`}
            >
              {step.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
