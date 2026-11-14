import type { PredictResponse, DecisionStatus } from "@/lib/api";
import PipelineVisual from "./PipelineVisual";

interface ResultsCardProps {
  result: PredictResponse;
  onReset: () => void;
}

const BANNER: Record<DecisionStatus, string> = {
  ACCEPT: "bg-sage-100 text-canopy-800",
  ABSTAIN_UNCERTAIN: "bg-glow-300/60 text-canopy-800",
  ABSTAIN_UNFAMILIAR: "bg-glow-300/60 text-canopy-800",
  NOT_A_LEAF: "bg-red-50 text-red-800",
  REJECT_QUALITY: "bg-red-50 text-red-800",
};

function label(result: PredictResponse) {
  switch (result.decision.status) {
    case "ACCEPT":
      return result.phyllotaxy.class;
    case "NOT_A_LEAF":
      return "Not a leaf";
    case "REJECT_QUALITY":
      return "Image unusable";
    default:
      return "No reliable answer";
  }
}

export default function ResultsCard({ result, onReset }: ResultsCardProps) {
  const confidencePct = Math.round(result.phyllotaxy.confidence * 100);
  const thresholdPct = Math.round(result.reliability.confidence_threshold * 100);
  const hasAqi = result.aqi.value !== null && result.aqi.value !== undefined;
  const probs = Object.entries(result.phyllotaxy.probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="w-full rounded-2xl border border-sage-200 bg-white p-6 sm:p-8">
      <PipelineVisual activeStep={5} />

      <div className={`mt-8 rounded-xl px-4 py-3 ${BANNER[result.decision.status]}`}>
        <p className="text-sm font-semibold">{result.decision.headline}</p>
        <ul className="mt-1 list-disc pl-5 text-xs leading-relaxed">
          {result.decision.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      </div>

      <div className="mt-8 grid gap-8 sm:grid-cols-2">
        <div>
          <p className="text-xs uppercase tracking-wide text-moss-400">Phyllotaxy</p>
          <p className="mt-1 font-display text-4xl text-canopy-800">{label(result)}</p>

          <div className="mt-4">
            <div className="flex items-center justify-between text-xs text-moss-500">
              <span>
                Calibrated confidence (top: {result.phyllotaxy.class}, threshold {thresholdPct}%)
              </span>
              <span className="font-semibold text-canopy-800">{confidencePct}%</span>
            </div>
            <div className="mt-1.5 h-2 w-full overflow-hidden rounded-full bg-sage-100">
              <div className="h-full rounded-full bg-moss-500" style={{ width: `${confidencePct}%` }} />
            </div>
          </div>

          <table className="mt-4 w-full text-xs text-moss-500">
            <tbody>
              {probs.map(([cls, p]) => (
                <tr key={cls}>
                  <td className="py-0.5">{cls}</td>
                  <td className="py-0.5 text-right tabular-nums">{(p * 100).toFixed(1)}%</td>
                </tr>
              ))}
              <tr>
                <td className="pt-2">Familiar to validated leaves</td>
                <td className="pt-2 text-right">{result.reliability.in_distribution ? "yes" : "no"}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div>
          {result.explanation && (
            <>
              <p className="text-xs uppercase tracking-wide text-moss-400">Where the model looked</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                className="mt-2 w-full rounded-xl"
                src={`data:image/png;base64,${result.explanation.heatmap_png_base64}`}
                alt="Attention rollout heatmap"
              />
              <p className="mt-1 text-[11px] leading-snug text-moss-400">{result.explanation.note}</p>
            </>
          )}
        </div>
      </div>

      <div className="mt-8">
        <p className="text-xs uppercase tracking-wide text-moss-400">Air quality estimate</p>
        {hasAqi ? (
          <p className="mt-1 font-display text-4xl text-canopy-800">
            {result.aqi.value}
            <span className="ml-2 text-base font-body text-moss-400">AQI · exploratory</span>
          </p>
        ) : (
          <p className="mt-1 text-sm font-medium text-canopy-800">
            No AQI shown ({result.aqi.status.replace("_", " ").toLowerCase()})
          </p>
        )}
        <p className="mt-1 text-sm text-moss-500">{result.aqi.message}</p>
        <p className="mt-1 text-xs text-moss-400">
          Evidence: {result.aqi.evidence.pairs} verified pairs · cross-validated R²{" "}
          {result.aqi.evidence.pooled_cv_r2} · permutation p = {result.aqi.evidence.permutation_p}
        </p>
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-sage-100 pt-4">
        <p className="text-xs text-moss-400">
          {result.model.classifier} · {result.model.device.toUpperCase()} · {result.model.latency_ms} ms ·{" "}
          {result.filename}
        </p>
        <button
          type="button"
          onClick={onReset}
          className="rounded-full border border-sage-200 px-4 py-1.5 text-xs font-semibold text-moss-600 transition-colors hover:border-moss-400"
        >
          Analyze another leaf
        </button>
      </div>
    </div>
  );
}
