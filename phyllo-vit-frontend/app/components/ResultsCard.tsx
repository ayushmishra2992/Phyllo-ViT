import type { PredictResponse } from "@/lib/api";
import PipelineVisual from "./PipelineVisual";

interface ResultsCardProps {
  result: PredictResponse;
  onReset: () => void;
}

export default function ResultsCard({ result, onReset }: ResultsCardProps) {
  const confidencePct = Math.round(result.phyllotaxy.confidence * 100);
  const hasAqi = result.aqi.value !== null && result.aqi.value !== undefined;

  return (
    <div className="w-full rounded-2xl border border-sage-200 bg-white p-6 sm:p-8">
      <PipelineVisual activeStep={5} />

      <div className="mt-8 grid gap-8 sm:grid-cols-2">
        <div>
          <p className="text-xs uppercase tracking-wide text-moss-400">
            Phyllotaxy class
          </p>
          <p className="mt-1 font-display text-4xl text-canopy-800">
            {result.phyllotaxy.class}
          </p>

          <div className="mt-4">
            <div className="flex items-center justify-between text-xs text-moss-500">
              <span>Confidence</span>
              <span className="font-semibold text-canopy-800">
                {confidencePct}%
              </span>
            </div>
            <div className="mt-1.5 h-2 w-full overflow-hidden rounded-full bg-sage-100">
              <div
                className="h-full rounded-full bg-moss-500"
                style={{ width: `${confidencePct}%` }}
              />
            </div>
          </div>
        </div>

        <div>
          <p className="text-xs uppercase tracking-wide text-moss-400">
            Air quality estimate
          </p>
          {hasAqi ? (
            <>
              <p className="mt-1 font-display text-4xl text-canopy-800">
                {result.aqi.value}
                <span className="ml-2 text-base font-body text-moss-400">
                  AQI
                </span>
              </p>
              <p className="mt-2 text-sm text-moss-500">{result.aqi.message}</p>
            </>
          ) : (
            <div className="mt-2 rounded-xl bg-sage-100 p-4">
              <p className="text-sm font-medium text-canopy-800">
                No AQI estimate available
              </p>
              <p className="mt-1 text-sm text-moss-500">{result.aqi.message}</p>
            </div>
          )}
        </div>
      </div>

      <p className="mt-6 rounded-xl bg-glow-300/40 px-4 py-3 text-xs leading-relaxed text-canopy-800">
        AQI prediction is a preliminary research estimate, currently
        validated only for Whorled-type leaves on a small dataset. Not a
        substitute for official air quality monitoring.
      </p>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-sage-100 pt-4">
        <p className="text-xs text-moss-400">
          Processed on {result.model.device.toUpperCase()} ·{" "}
          {result.model.embedding_dimensions}-D embedding · {result.filename}
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
