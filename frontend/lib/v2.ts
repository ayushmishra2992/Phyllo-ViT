export type ModuleStatus =
  | "AVAILABLE"
  | "EXPERIMENTAL"
  | "USER_INPUT_REQUIRED"
  | "UNAVAILABLE"
  | "NOT_SUPPORTED"
  | "DISABLED";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Section = { status: ModuleStatus; reason: string } & Record<string, any>;

export interface AnalyzeResponse {
  version: string;
  filename: string;
  image_quality: Section;
  phyllotaxy: Section;
  species: Section;
  morphology: Section;
  leaf_health: Section;
  dust_deposition: Section;
  leaf_based_aqi: Section;
  ambient_environment: Section;
  ftir: Section;
  biological: Section;
  separation_notice: string;
  latency_ms: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export class V2Error extends Error {}

export interface AnalyzeInput {
  file: File;
  lat?: string;
  lon?: string;
  collectedAt?: string;
  ftir?: File | null;
  ftirUnits?: string;
  ftirWavenumberScale?: string;
}

export async function analyze(input: AnalyzeInput): Promise<AnalyzeResponse> {
  const fd = new FormData();
  fd.append("file", input.file);
  if (input.lat?.trim()) fd.append("lat", input.lat.trim());
  if (input.lon?.trim()) fd.append("lon", input.lon.trim());
  if (input.collectedAt?.trim()) fd.append("collected_at", input.collectedAt.trim());
  if (input.ftir) {
    fd.append("ftir_file", input.ftir);
    fd.append("ftir_y_units", input.ftirUnits ?? "absorbance");
    if (input.ftirWavenumberScale?.trim()) fd.append("ftir_wavenumber_scale", input.ftirWavenumberScale.trim());
  }
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/v2/analyze`, { method: "POST", body: fd });
  } catch {
    throw new V2Error(
      `Could not reach the Phyllo-ViT 2.0 backend at ${API_BASE_URL}. Start it with RUN_V2.bat (python -m uvicorn api.v2_app:app).`
    );
  }
  if (!res.ok) {
    let detail = "";
    try {
      const b = await res.json();
      detail = typeof b?.detail === "string" ? b.detail : JSON.stringify(b?.detail ?? "");
    } catch {
      /* ignore */
    }
    throw new V2Error(detail || `The backend returned an error (${res.status}).`);
  }
  return (await res.json()) as AnalyzeResponse;
}

export function reportHtml(r: AnalyzeResponse): string {
  const esc = (s: unknown) =>
    String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c] as string));
  const rows = (
    [
      ["Image quality", r.image_quality],
      ["Phyllotaxy", r.phyllotaxy],
      ["Species", r.species],
      ["Morphology", r.morphology],
      ["Visible condition", r.leaf_health],
      ["Dust / deposition", r.dust_deposition],
      ["Leaf-based AQI", r.leaf_based_aqi],
      ["Ambient environment", r.ambient_environment],
      ["FTIR", r.ftir],
      ["Biological", r.biological],
    ] as [string, Section][]
  )
    .map(([n, s]) => `<tr><td>${esc(n)}</td><td>${esc(s.status)}</td><td>${esc(s.reason)}</td></tr>`)
    .join("");
  return `<!doctype html><meta charset="utf-8"><title>Phyllo-ViT 2.0 report</title>
<style>body{font:14px system-ui;margin:2rem;max-width:900px}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}table{border-collapse:collapse;width:100%}</style>
<h1>Phyllo-ViT 2.0 analysis report</h1><p>File: ${esc(r.filename)} · generated ${new Date().toISOString()} · backend v${esc(r.version)}</p>
<table><tr><th>Module</th><th>Status</th><th>Explanation</th></tr>${rows}</table>
<p><b>Phyllotaxy label:</b> ${esc(r.phyllotaxy.label)} (calibrated confidence ${esc(
    Math.round((r.phyllotaxy.calibrated_confidence ?? 0) * 100)
  )}%)</p><p>${esc(r.separation_notice)}</p>
<p>Unsupported or unavailable modules are shown as such; no values are invented. Full data: attached JSON.</p>`;
}
