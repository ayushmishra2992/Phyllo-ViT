export type PhyllotaxyClass = "Alternate" | "Opposite" | "Whorled";

export interface PredictResponse {
  filename: string;
  phyllotaxy: {
    class: PhyllotaxyClass;
    confidence: number;
  };
  aqi: {
    value: number | null;
    message: string;
  };
  model: {
    device: string;
    embedding_dimensions: number;
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
    response = await fetch(`${API_BASE_URL}/predict`, {
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
