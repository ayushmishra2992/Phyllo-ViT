interface ErrorStateProps {
  message: string;
  onRetry: () => void;
}

export default function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="w-full rounded-2xl border border-red-200 bg-red-50 p-6 text-center sm:p-8">
      <p className="font-display text-lg text-red-800">
        The prediction didn&apos;t go through
      </p>
      <p className="mx-auto mt-2 max-w-sm text-sm text-red-700/80">
        {message}
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-5 rounded-full border border-red-300 bg-white px-5 py-2 text-sm font-semibold text-red-700 transition-colors hover:bg-red-100"
      >
        Try again
      </button>
    </div>
  );
}
