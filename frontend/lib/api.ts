export type PhyllotaxyClass = "Alternate" | "Opposite" | "Whorled" | "Not_a_leaf";

export type DecisionStatus =
  | "ACCEPT"
  | "ABSTAIN_UNCERTAIN"
  | "ABSTAIN_UNFAMILIAR"
  | "NOT_A_LEAF"
  | "REJECT_QUALITY";

export type AqiStatus = "NOT_APPLICABLE" | "WITHHELD" | "EXPLORATORY";

// Response of the Phyllo-ViT v3 backend (POST /predict). The v2 fields
// (filename, phyllotaxy.class/confidence, aqi.value/message, model.*) are unchanged.
export interface PredictResponse {
  filename: string;
  decision: {
    status: DecisionStatus;
    accepted: boolean;
    headline: string;
    reasons: string[];
  };
  phyllotaxy: {
    class: PhyllotaxyClass;
    accepted: boolean;
    confidence: number; // calibrated
    raw_confidence: number;
    probabilities: Record<PhyllotaxyClass, number>;
  };
  reliability: {
    temperature: number;
    confidence_threshold: number;
    mahalanobis_score: number;
    mahalanobis_threshold: number;
    knn_similarity: number;
    knn_threshold: number;
    in_distribution: boolean;
  };
  quality: {
    passed: boolean;
    sharpness: number;
    noise: number;
    issues: string[];
    warnings: string[];
  };
  explanation: {
    method: string;
    note: string;
    heatmap_png_base64: string;
  } | null;
  aqi: {
    value: number | null;
    status: AqiStatus;
    message: string;
    exploratory_value?: number;
    exploratory_warning?: string;
    evidence: { pairs: number; pooled_cv_r2: number; permutation_p: number; supported: boolean };
  };
  model: {
    device: string;
    classifier: string;
    embedding_dimensions: number;
    latency_ms: number;
  };
}

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function predict(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/predict?explain=true`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new ApiError(
      "Could not reach the Phyllo-ViT backend. Make sure the FastAPI server is running on " +
        API_BASE_URL +
        "."
    );
  }

  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = body?.detail ?? "";
    } catch {
      // ignore unparsable error body
    }
    throw new ApiError(
      detail || `The backend returned an error (${response.status}).`,
      response.status
    );
  }

  return (await response.json()) as PredictResponse;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
