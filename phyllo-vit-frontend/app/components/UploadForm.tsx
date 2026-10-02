"use client";

import {
  useCallback,
  useRef,
  useState,
  type DragEvent,
  type KeyboardEvent,
} from "react";

const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10MB

interface UploadFormProps {
  onSubmit: (file: File) => void;
  disabled: boolean;
}

export default function UploadForm({ onSubmit, disabled }: UploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const acceptFile = useCallback((candidate: File | undefined) => {
    if (!candidate) return;
    if (!candidate.type.startsWith("image/")) {
      setValidationError("Please choose an image file (JPG, PNG, etc.).");
      return;
    }
    if (candidate.size > MAX_SIZE_BYTES) {
      setValidationError("That image is over 10MB — please choose a smaller file.");
      return;
    }
    setValidationError(null);
    setFile(candidate);
    setPreviewUrl(URL.createObjectURL(candidate));
  }, []);

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    acceptFile(e.dataTransfer.files?.[0]);
  };

  return (
    <div className="w-full">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e: KeyboardEvent<HTMLDivElement>) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        className={`flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
          isDragging
            ? "border-moss-500 bg-sage-100"
            : "border-sage-200 bg-white hover:border-moss-400"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => acceptFile(e.target.files?.[0])}
        />
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="Selected leaf preview"
            className="max-h-48 rounded-lg object-contain"
          />
        ) : (
          <>
            <p className="font-display text-lg text-canopy-800">
              Drop a leaf photo here
            </p>
            <p className="mt-1 text-sm text-moss-500">
              or click to browse · JPG or PNG, up to 10MB
            </p>
          </>
        )}
      </div>

      {validationError && (
        <p className="mt-2 text-sm text-red-600">{validationError}</p>
      )}

      <div className="mt-5 flex items-center justify-between gap-3">
        {file && !validationError ? (
          <p className="truncate text-sm text-moss-500">{file.name}</p>
        ) : (
          <span />
        )}
        <button
          type="button"
          disabled={!file || !!validationError || disabled}
          onClick={() => file && onSubmit(file)}
          className="shrink-0 rounded-full bg-moss-600 px-6 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-moss-500 disabled:cursor-not-allowed disabled:bg-sage-200 disabled:text-moss-400"
        >
          {disabled ? "Analyzing…" : "Predict"}
        </button>
      </div>
    </div>
  );
}
