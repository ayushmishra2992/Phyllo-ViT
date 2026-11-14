import PipelineVisual from "./PipelineVisual";

interface LoadingStateProps {
  activeStep: number;
}

export default function LoadingState({ activeStep }: LoadingStateProps) {
  return (
    <div className="w-full rounded-2xl border border-sage-200 bg-white p-6 sm:p-8">
      <PipelineVisual activeStep={activeStep} />
      <div className="mt-8 space-y-3">
        <div className="h-4 w-1/3 animate-pulse rounded bg-sage-100" />
        <div className="h-8 w-2/3 animate-pulse rounded bg-sage-100" />
        <div className="h-4 w-1/2 animate-pulse rounded bg-sage-100" />
      </div>
    </div>
  );
}
