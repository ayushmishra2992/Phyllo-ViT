"use client";

import { useRef, useState } from "react";
import { analyze, reportHtml, V2Error, type AnalyzeResponse, type ModuleStatus, type Section } from "@/lib/v2";

const BADGE: Record<ModuleStatus, { text: string; cls: string }> = {
  AVAILABLE: { text: "Available", cls: "bg-sage-100 text-canopy-800" },
  EXPERIMENTAL: { text: "Experimental", cls: "bg-glow-300/60 text-canopy-800" },
  USER_INPUT_REQUIRED: { text: "Input required", cls: "bg-blue-50 text-blue-800" },
  UNAVAILABLE: { text: "Unavailable", cls: "bg-red-50 text-red-800" },
  NOT_SUPPORTED: { text: "Not yet supported", cls: "bg-gray-100 text-gray-700" },
  DISABLED: { text: "Disabled", cls: "bg-gray-100 text-gray-700" },
};

function Badge({ s }: { s: ModuleStatus }) {
  const b = BADGE[s];
  return <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${b.cls}`}>{b.text}</span>;
}

function Card({ title, sec, children }: { title: string; sec: Section; children?: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-sage-200 bg-white p-5">
      <div className="flex items-center justify-between gap-3">
        <h3 className="font-display text-lg text-canopy-800">{title}</h3>
        <Badge s={sec.status} />
      </div>
      <p className="mt-2 text-sm text-moss-500">{sec.reason}</p>
      {children && <div className="mt-3">{children}</div>}
    </section>
  );
}

function Spectrum({ wn, ab }: { wn: number[]; ab: number[] }) {
  const w = 560, h = 180, p = 28;
  const xmin = Math.min(...wn), xmax = Math.max(...wn);
  const ymin = Math.min(...ab), ymax = Math.max(...ab);
  const X = (v: number) => p + ((xmax - v) / (xmax - xmin)) * (w - 2 * p); // IR convention: high wavenumber on the left
  const Y = (v: number) => h - p - ((v - ymin) / (ymax - ymin || 1)) * (h - 2 * p);
  const d = wn.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(ab[i]).toFixed(1)}`).join("");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label="FTIR spectrum">
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.2" className="text-moss-600" />
      <line x1={p} y1={h - p} x2={w - p} y2={h - p} stroke="currentColor" className="text-sage-200" />
      <text x={p} y={h - 8} fontSize="10" className="fill-current text-moss-500">{Math.round(xmax)} cm⁻¹</text>
      <text x={w - p} y={h - 8} fontSize="10" textAnchor="end" className="fill-current text-moss-500">{Math.round(xmin)} cm⁻¹</text>
      <text x={p} y={12} fontSize="10" className="fill-current text-moss-500">absorbance</text>
    </svg>
  );
}

function download(name: string, mime: string, body: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([body], { type: mime }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

export default function V2Page() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [ftir, setFtir] = useState<File | null>(null);
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [when, setWhen] = useState("");
  const [units, setUnits] = useState("absorbance");
  const [scale, setScale] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<AnalyzeResponse | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const pick = (f?: File) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) return setErr("Please choose an image file (JPG, PNG, WEBP or BMP).");
    if (f.size > 10 * 1024 * 1024) return setErr("That image is over 10 MB.");
    setErr(""); setFile(f); setPreview(URL.createObjectURL(f)); setRes(null);
  };

  const run = async () => {
    if (!file) return;
    setBusy(true); setErr("");
    try {
      setRes(await analyze({ file, lat, lon, collectedAt: when, ftir, ftirUnits: units, ftirWavenumberScale: scale }));
    } catch (e) {
      setErr(e instanceof V2Error ? e.message : "Something unexpected went wrong.");
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setFile(null); setPreview(null); setFtir(null); setRes(null); setErr("");
    if (inputRef.current) inputRef.current.value = "";
  };

  const ph = res?.phyllotaxy;
  const probs = ph ? Object.entries(ph.probabilities as Record<string, number>).sort((a, b) => b[1] - a[1]) : [];
  const env = res?.ambient_environment;
  const st = env?.nearest_station;

  return (
    <main className="mx-auto max-w-5xl px-5 py-10">
      <h1 className="font-display text-4xl text-canopy-800">Phyllo-ViT 2.0 — leaf analysis</h1>
      <p className="mt-2 max-w-3xl text-sm text-moss-500">
        Every section shows its status. Unsupported or unavailable measurements are labelled, never filled in. A leaf photograph cannot measure AQI,
        PM2.5, chemical composition or microbes; ambient readings come from monitoring stations and are shown separately.
      </p>

      <div className="mt-6 grid gap-5 md:grid-cols-2">
        <div className="rounded-2xl border border-sage-200 bg-white p-5">
          <h2 className="font-display text-lg text-canopy-800">1 · Leaf / stem image</h2>
          <div
            role="button" tabIndex={0} onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); pick(e.dataTransfer.files?.[0]); }}
            className="mt-3 flex min-h-48 cursor-pointer items-center justify-center rounded-xl border-2 border-dashed border-sage-200 p-4 text-center hover:border-moss-400"
          >
            <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="Selected leaf preview" className="max-h-44 rounded-lg object-contain" />
            ) : (
              <span className="text-sm text-moss-500">Drop an image or click to browse · up to 10 MB</span>
            )}
          </div>
        </div>

        <div className="rounded-2xl border border-sage-200 bg-white p-5">
          <h2 className="font-display text-lg text-canopy-800">2 · Optional context</h2>
          <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
            <label className="text-moss-500">Latitude<input value={lat} onChange={(e) => setLat(e.target.value)} placeholder="e.g. 28.46" className="mt-1 w-full rounded-lg border border-sage-200 px-2 py-1.5" /></label>
            <label className="text-moss-500">Longitude<input value={lon} onChange={(e) => setLon(e.target.value)} placeholder="e.g. 77.03" className="mt-1 w-full rounded-lg border border-sage-200 px-2 py-1.5" /></label>
            <label className="col-span-2 text-moss-500">Collection date/time (recorded in the report; ambient data is the latest available)
              <input value={when} onChange={(e) => setWhen(e.target.value)} placeholder="2026-03-01 17:00" className="mt-1 w-full rounded-lg border border-sage-200 px-2 py-1.5" /></label>
            <label className="col-span-2 text-moss-500">FTIR spectrum (CSV/TXT, XLSX, JCAMP-DX)
              <input type="file" accept=".csv,.tsv,.txt,.dat,.xlsx,.jdx,.dx" onChange={(e) => setFtir(e.target.files?.[0] ?? null)} className="mt-1 w-full text-xs" /></label>
            <label className="text-moss-500">FTIR y-axis
              <select value={units} onChange={(e) => setUnits(e.target.value)} className="mt-1 w-full rounded-lg border border-sage-200 px-2 py-1.5">
                <option value="absorbance">absorbance</option><option value="transmittance">transmittance</option></select></label>
            <label className="text-moss-500">Wavenumber divisor (optional)<input value={scale} onChange={(e) => setScale(e.target.value)} placeholder="e.g. 1000" className="mt-1 w-full rounded-lg border border-sage-200 px-2 py-1.5" /></label>
          </div>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <button type="button" disabled={!file || busy} onClick={run}
          className="rounded-full bg-moss-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-moss-500 disabled:cursor-not-allowed disabled:bg-sage-200 disabled:text-moss-400">
          {busy ? "Analysing…" : "Analyse"}
        </button>
        <button type="button" onClick={reset} className="rounded-full border border-sage-200 px-4 py-2 text-sm font-semibold text-moss-600 hover:border-moss-400">Reset</button>
        {err && <p role="alert" className="text-sm text-red-600">{err}</p>}
      </div>
      {!res && !busy && !err && <p className="mt-8 text-sm text-moss-400">Upload an image and press Analyse. Nothing is shown until the backend returns real results.</p>}

      {res && ph && (
        <div className="mt-8 grid gap-4 md:grid-cols-2">
          <Card title="Image quality" sec={res.image_quality}>
            <p className="text-xs text-moss-500">sharpness {res.image_quality.sharpness} · noise {res.image_quality.noise}</p>
            {(res.image_quality.warnings as string[] | undefined)?.map((w) => <p key={w} className="text-xs text-moss-400">⚠ {w}</p>)}
          </Card>

          <Card title="Phyllotaxy" sec={ph}>
            <p className="font-display text-3xl text-canopy-800">{ph.label}</p>
            <p className="text-xs text-moss-500">Calibrated confidence {Math.round((ph.calibrated_confidence ?? 0) * 100)}% · decision {ph.decision_status}</p>
            <table className="mt-2 w-full text-xs text-moss-500"><tbody>
              {probs.map(([c, p]) => (<tr key={c}><td>{c}</td><td className="text-right tabular-nums">{(p * 100).toFixed(1)}%</td></tr>))}
            </tbody></table>
            <p className="mt-2 text-xs text-moss-400">{ph.context_note}</p>
            <p className="mt-1 text-xs text-moss-400">{ph.validation_note}</p>
            <p className="mt-1 text-xs text-moss-400">Spiral: not supported (no labelled data).</p>
          </Card>

          <Card title="Species identification" sec={res.species} />
          <Card title="Leaf morphology" sec={res.morphology}>
            {res.morphology.status === "AVAILABLE" && (
              <ul className="text-xs text-moss-500">
                <li>Leaf area fraction of frame: {(res.morphology.leaf_area_fraction * 100).toFixed(1)}%</li>
                <li>Aspect ratio {res.morphology.aspect_ratio} · solidity {res.morphology.solidity} · eccentricity {res.morphology.eccentricity}</li>
                {res.morphology.touches_image_border && <li>Region touches the image border (leaf may be cropped).</li>}
                <li className="mt-1 text-moss-400">{res.morphology.caveat}</li>
              </ul>)}
          </Card>

          <Card title="Visible leaf condition" sec={res.leaf_health}>
            {res.leaf_health.status === "EXPERIMENTAL" && (
              <>
                <table className="w-full text-xs text-moss-500"><tbody>
                  {Object.entries(res.leaf_health.colour_fractions as Record<string, number>).map(([k, v]) => (
                    <tr key={k}><td>{k.replace("_", " ")}</td><td className="text-right tabular-nums">{(v * 100).toFixed(1)}%</td></tr>))}
                </tbody></table>
                {(res.leaf_health.flags as string[]).length === 0
                  ? <p className="mt-1 text-xs text-moss-400">No colour flags raised.</p>
                  : (res.leaf_health.flags as string[]).map((f) => <p key={f} className="mt-1 text-xs text-moss-500">⚑ {f}</p>)}
              </>)}
          </Card>

          <Card title="Dust / surface deposition" sec={res.dust_deposition} />
          <Card title="Leaf-image AQI estimate" sec={res.leaf_based_aqi} />

          <div className="md:col-span-2">
            <Card title="Ambient air quality (monitoring station)" sec={env!}>
              {env!.status === "AVAILABLE" && st && (
                <div className="text-xs text-moss-500">
                  <p className="font-semibold text-canopy-800">{st.name} · {st.distance_km} km away · provider {st.provider ?? "unknown"} · licence {(st.licences as string[]).join(", ") || "not stated"}</p>
                  <table className="mt-2 w-full"><thead><tr className="text-left text-moss-400"><th>Pollutant</th><th>Value</th><th>Unit</th><th>Time (UTC)</th></tr></thead><tbody>
                    {(st.readings as { parameter: string; value: number; unit: string; timestamp_utc: string }[]).map((r) => (
                      <tr key={r.parameter}><td>{r.parameter}</td><td className="tabular-nums">{r.value}</td><td>{r.unit}</td><td>{r.timestamp_utc}</td></tr>))}
                  </tbody></table>
                  {env!.indicative_index_from_latest && <p className="mt-2">Indicative index from latest values: <b>{env!.indicative_index_from_latest.aqi}</b> ({env!.indicative_index_from_latest.category}). {env!.indicative_index_from_latest.note}</p>}
                  <p className="mt-1 text-moss-400">{env!.representativeness_note} {env!.time_note}</p>
                  <p className="mt-1 text-moss-400">Source: {env!.source} · retrieved {env!.retrieved_utc}</p>
                </div>)}
            </Card>
          </div>

          <div className="md:col-span-2">
            <Card title="FTIR analysis" sec={res.ftir}>
              {res.ftir.status === "AVAILABLE" && (
                <>
                  <Spectrum wn={res.ftir.spectrum.wavenumber} ab={res.ftir.spectrum.absorbance} />
                  <p className="text-xs text-moss-500">{res.ftir.n_points} points · {res.ftir.range_cm1[0]}–{res.ftir.range_cm1[1]} cm⁻¹ · {res.ftir.n_peaks} peaks</p>
                  <table className="mt-2 w-full text-xs text-moss-500"><thead><tr className="text-left text-moss-400"><th>cm⁻¹</th><th>Abs.</th><th>Candidate group ranges (indicative)</th></tr></thead><tbody>
                    {(res.ftir.peaks as { "wavenumber_cm-1": number; absorbance: number; candidate_groups: { assignment: string }[] }[]).slice(0, 12).map((p) => (
                      <tr key={p["wavenumber_cm-1"]}><td className="tabular-nums">{p["wavenumber_cm-1"]}</td><td className="tabular-nums">{p.absorbance}</td>
                        <td>{p.candidate_groups.map((c) => c.assignment).join("; ") || "—"}</td></tr>))}
                  </tbody></table>
                  <p className="mt-2 text-xs text-moss-400">{res.ftir.caveat}</p>
                </>)}
            </Card>
          </div>

          <Card title="Biological contamination" sec={res.biological} />
          <section className="rounded-2xl border border-sage-200 bg-white p-5">
            <h3 className="font-display text-lg text-canopy-800">Report</h3>
            <p className="mt-2 text-xs text-moss-500">{res.separation_notice}</p>
            <p className="mt-1 text-xs text-moss-400">{res.filename} · {res.latency_ms} ms · backend v{res.version}</p>
            <div className="mt-3 flex gap-2">
              <button type="button" onClick={() => download("phyllo-vit-report.json", "application/json", JSON.stringify(res, null, 2))} className="rounded-full border border-sage-200 px-4 py-1.5 text-xs font-semibold text-moss-600 hover:border-moss-400">Download JSON</button>
              <button type="button" onClick={() => download("phyllo-vit-report.html", "text/html", reportHtml(res))} className="rounded-full border border-sage-200 px-4 py-1.5 text-xs font-semibold text-moss-600 hover:border-moss-400">Download report (HTML)</button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
