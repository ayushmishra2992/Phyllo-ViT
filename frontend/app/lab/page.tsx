"use client";

import { useEffect, useRef, useState } from "react";
import { predict, ApiError, type PredictResponse } from "@/lib/api";
import UploadForm from "@/app/components/UploadForm";
import LoadingState from "@/app/components/LoadingState";
import ResultsCard from "@/app/components/ResultsCard";
import ErrorState from "@/app/components/ErrorState";

type Status = "idle" | "loading" | "success" | "error";

const FEATURES = [
  {
    title: "Upload a leaf",
    body: "Any roadside or garden leaf, photographed clearly against a plain background.",
  },
  {
    title: "ViT reads the pattern",
    body: "A fine-tuned Vision Transformer encodes the leaf into a 768-dimension representation.",
  },
  {
    title: "Reliability is checked",
    body: "Calibrated confidence and distance to validated leaves decide whether the model may answer at all.",
  },
  {
    title: "Evidence gates the AQI",
    body: "An AQI number is shown only if cross-validated evidence supports it — currently it is withheld.",
  },
];

export default function LabPage() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [activeStep, setActiveStep] = useState(0);
  const stepTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleSubmit = async (file: File) => {
    setStatus("loading");
    setActiveStep(0);

    // Step the pipeline visual forward while the request is in flight
    stepTimer.current = setInterval(() => {
      setActiveStep((prev) => (prev < 3 ? prev + 1 : prev));
    }, 500);

    try {
      const response = await predict(file);

      if (stepTimer.current) clearInterval(stepTimer.current);

      setActiveStep(5);
      setResult(response);
      setStatus("success");
    } catch (err) {
      if (stepTimer.current) clearInterval(stepTimer.current);

      const message =
        err instanceof ApiError
          ? err.message
          : "Something unexpected went wrong. Please try again.";

      setErrorMessage(message);
      setStatus("error");
    }
  };

  const handleReset = () => {
    setStatus("idle");
    setResult(null);
    setErrorMessage("");
    setActiveStep(0);
  };

  useEffect(() => {
    return () => {
      if (stepTimer.current) clearInterval(stepTimer.current);
    };
  }, []);

  return (
    <main className="min-h-screen bg-paper">

      {/* Main heading */}
      <section className="mx-auto max-w-3xl px-6 pt-14 pb-8 text-center sm:pt-16 sm:pb-10">
        <h1 className="font-display text-4xl font-bold leading-tight tracking-tight text-canopy-800 sm:text-5xl md:text-6xl">
          Decode your leaf
        </h1>

        <p className="mx-auto mt-5 max-w-xl text-base font-medium leading-relaxed text-moss-500 sm:text-lg">
          Upload a photo of a leaf. We classify its phyllotaxy and, where the
          data supports it, estimate the air quality around it.
        </p>
      </section>

      {/* How the system works */}
      <section className="mx-auto max-w-5xl px-6 pt-5 pb-8 sm:pt-6 sm:pb-10">
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-4 lg:gap-x-10 lg:gap-y-6">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="mx-auto max-w-xs text-center"
            >
              <p className="font-display text-sm font-bold leading-snug text-canopy-800 sm:text-base">
                {f.title}
              </p>

              <p className="mt-2 text-sm font-medium leading-relaxed text-moss-500">
                {f.body}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Upload / loading / results */}
      <section className="mx-auto max-w-2xl px-6 pt-3 pb-20 sm:pt-5 sm:pb-24">
        {status === "idle" && (
          <UploadForm onSubmit={handleSubmit} disabled={false} />
        )}

        {status === "loading" && (
          <LoadingState activeStep={activeStep} />
        )}

        {status === "success" && result && (
          <ResultsCard
            result={result}
            onReset={handleReset}
          />
        )}

        {status === "error" && (
          <ErrorState
            message={errorMessage}
            onRetry={handleReset}
          />
        )}
      </section>
    </main>
  );
}