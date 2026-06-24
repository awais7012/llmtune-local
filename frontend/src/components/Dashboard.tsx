import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchConfig,
  fetchModels,
  fetchCachedModels,
  fetchTrainedModels,
  jobWebSocket,
  pollJob,
  startTraining,
  stopJob,
  validateDataset,
  inferModel,
  pushModel,
  exportGguf,
  type AppConfig,
  type DatasetValidation,
  type TrainedModel,
} from "../api";
import type { ModelInfo, WizardState } from "../types";
import { defaultWizardState } from "../types";
import LossChart from "./LossChart";
import KagglePanel from "./KagglePanel";
import "../dashboard.css";

// ── Constants ────────────────────────────────────────────────────────────────
const CONFIG_KEY = "llmtune-studio-config";
const LR_VALUES = [5e-6, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3];
const PRESETS = [
  { label: "Fast test", icon: "⚡", epochs: 1, batchSize: 4, learningRate: 5e-5, gradAccum: 2, loraR: 8 },
  { label: "Balanced", icon: "⚖", epochs: 3, batchSize: 2, learningRate: 2e-4, gradAccum: 4, loraR: 16 },
  { label: "Best quality", icon: "🏆", epochs: 5, batchSize: 1, learningRate: 1e-4, gradAccum: 8, loraR: 32 },
];
const PHASE_LABELS: Record<string, string> = {
  loading: "Loading model", downloading: "Downloading", dataset: "Preparing dataset",
  training: "Training", done: "Complete", error: "Failed", stopped: "Stopped",
};
type Tab = "home" | "studio" | "train" | "models";

// ── Helpers ──────────────────────────────────────────────────────────────────
function loadSavedConfig(): Partial<WizardState> | null {
  try { const raw = localStorage.getItem(CONFIG_KEY); return raw ? JSON.parse(raw) : null; } catch { return null; }
}
function isHfId(p: string): boolean {
  const s = p.trim();
  if (!s || s.startsWith("/") || s.startsWith("~") || s.startsWith(".")) return false;
  const parts = s.split("/");
  return parts.length === 2 && parts.every(Boolean);
}
function formatLR(v: number): string {
  if (v === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(v)));
  const m = v / Math.pow(10, exp);
  if (Math.abs(m - 1) < 0.01) return `1e${exp}`;
  if (Math.abs(m - 2) < 0.01) return `2e${exp}`;
  if (Math.abs(m - 5) < 0.01) return `5e${exp}`;
  return v.toExponential(1);
}
function fmtEta(s: number): string {
  if (!isFinite(s) || s <= 0) return "—";
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function methodLabel(s: WizardState): string {
  if (s.mode === "cnn") return ({ feature_extraction: "Feature extraction", full: "Full fine-tune", lora: "LoRA" } as const)[s.trainingMode] || "Feature extraction";
  if (s.quantization === "4bit") return "QLoRA (4-bit)";
  if (s.quantization === "8bit") return "QLoRA (8-bit)";
  if (s.trainingMode === "full") return "Full fine-tune";
  return "LoRA (16-bit)";
}
function estimateVram(method: string, batch: number, modelGb = 7): number {
  const base = method.includes("4-bit") ? modelGb * 0.6 : method.includes("8-bit") ? modelGb * 0.8 : method.includes("Full") ? modelGb * 2.2 : modelGb * 1.4;
  return base + (batch - 1) * 0.3;
}
function formatBytes(n: number): string {
  if (!n) return "0 MB";
  const mb = n / 1048576;
  return mb < 1024 ? `${mb.toFixed(mb < 10 ? 1 : 0)} MB` : `${(mb / 1024).toFixed(1)} GB`;
}
function fmtDate(ts: number): string {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

interface Props {
  skipAuth: boolean;
  onLogout?: () => void;
}

// ── Component ────────────────────────────────────────────────────────────────
export default function Dashboard({ skipAuth, onLogout }: Props) {
  const [tab, setTab] = useState<Tab>("home");
  const [appConfig, setAppConfig] = useState<AppConfig | null>(null);
  const [state, setState] = useState<WizardState>(() => ({ ...defaultWizardState(), ...loadSavedConfig() }));

  const [popular, setPopular] = useState<{ llm: ModelInfo[]; cnn: ModelInfo[] }>({ llm: [], cnn: [] });
  const [cachedModels, setCachedModels] = useState<string[]>([]);
  const [trained, setTrained] = useState<TrainedModel[]>([]);
  const [modelQuery, setModelQuery] = useState(state.modelId || "");
  const [datasetStatus, setDatasetStatus] = useState<DatasetValidation | null>(null);
  const [homeSearch, setHomeSearch] = useState("");
  const [homeFilter, setHomeFilter] = useState<"all" | "llm" | "cnn">("all");
  const [dsSource, setDsSource] = useState<"local" | "kaggle">("local");
  const [activePreset, setActivePreset] = useState<number | null>(1);

  // Training state
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState("");
  const [training, setTraining] = useState(false);
  const [phase, setPhase] = useState("");
  const [step, setStep] = useState(0);
  const [total, setTotal] = useState(0);
  const [loss, setLoss] = useState(-1);
  const [lossHistory, setLossHistory] = useState<{ step: number; loss: number }[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [banner, setBanner] = useState<{ type: "success" | "error" | "warn"; text: string } | null>(null);
  const [finished, setFinished] = useState(false);
  const [connected, setConnected] = useState(false);
  const [startTime, setStartTime] = useState(0);
  const [eta, setEta] = useState(-1);
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  const patch = useCallback((p: Partial<WizardState>) => {
    setState((prev) => ({ ...prev, ...p }));
    setError("");
  }, []);

  const refreshTrained = useCallback(() => { fetchTrainedModels().then(setTrained); }, []);

  // Persist config (incl. HF token & model/dataset choices) so they survive restarts.
  useEffect(() => {
    const t = setTimeout(() => { try { localStorage.setItem(CONFIG_KEY, JSON.stringify(state)); } catch { /* ignore */ } }, 300);
    return () => clearTimeout(t);
  }, [state]);

  useEffect(() => {
    fetchConfig().then(setAppConfig);
    fetchModels().then(setPopular).catch(() => {});
    fetchCachedModels().then(setCachedModels);
    refreshTrained();
  }, [refreshTrained]);

  useEffect(() => {
    if (appConfig && !state.outputDir) {
      patch({ outputDir: state.mode === "cnn" ? appConfig.default_cnn_output : appConfig.default_llm_output });
    }
  }, [appConfig, state.mode, state.outputDir, patch]);

  // Dataset validation (local files only)
  useEffect(() => {
    if (!state.datasetPath || isHfId(state.datasetPath)) { setDatasetStatus(null); return; }
    const t = setTimeout(async () => setDatasetStatus(await validateDataset(state.datasetPath)), 600);
    return () => clearTimeout(t);
  }, [state.datasetPath]);

  // ETA
  useEffect(() => {
    if (training && step > 0 && total > 0 && startTime > 0) {
      const elapsed = (Date.now() - startTime) / 1000;
      setEta((total - step) / (step / elapsed));
    }
  }, [step, total, training, startTime]);

  function applyJob(data: {
    status?: string; step?: number; total_steps?: number; loss?: number;
    phase?: string; logs?: string[]; output_path?: string; error?: string;
  }) {
    if (data.phase) setPhase(data.phase);
    if (data.step !== undefined) setStep(data.step);
    if (data.total_steps !== undefined) setTotal(data.total_steps);
    if (data.loss !== undefined && data.loss >= 0) {
      setLoss(data.loss);
      setLossHistory((prev) => {
        const s = data.step ?? prev.length;
        if (prev.length && prev[prev.length - 1].step === s) return prev;
        return [...prev.slice(-299), { step: s, loss: data.loss! }];
      });
    }
    if (data.logs?.length) setLogs(data.logs);
    if (data.status === "completed" || data.output_path) {
      setBanner({ type: "success", text: `✓ Saved → ${data.output_path}` });
      setPhase("done"); setFinished(true); setTraining(false); refreshTrained();
    } else if (data.status === "failed" || data.error) {
      setBanner({ type: "error", text: data.error || "Training failed" });
      setPhase("error"); setFinished(true); setTraining(false);
    } else if (data.status === "stopped") {
      setBanner({ type: "warn", text: "Training stopped." });
      setPhase("stopped"); setFinished(true); setTraining(false);
    }
  }

  useEffect(() => {
    if (!jobId || finished) return;
    let alive = true;
    const poll = async () => { if (!alive) return; try { applyJob(await pollJob(jobId)); } catch { /* ignore */ } };
    poll();
    const id = setInterval(poll, 2000);
    return () => { alive = false; clearInterval(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, finished]);

  useEffect(() => {
    if (!jobId) return;
    const ws = jobWebSocket(jobId);
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.type === "snapshot") applyJob({ status: data.status, step: data.step, total_steps: data.total, loss: data.loss, phase: data.phase, logs: data.logs, output_path: data.output_path, error: data.error });
      if (data.type === "log") {
        setLogs((prev) => [...prev.slice(-499), data.message]);
        const l = data.message.toLowerCase();
        if (l.includes("preprocessing") || l.includes("loading dataset")) setPhase("dataset");
        else if (l.includes("starting training")) { setPhase("training"); setDownloadProgress(null); }
        if (l.includes("download")) {
          const m = data.message.match(/(\d+)%/);
          if (m) { const pv = parseInt(m[1]); setDownloadProgress(pv); if (pv < 100) setPhase("downloading"); }
        }
      }
      if (data.type === "progress") {
        setStep(data.step); setTotal(data.total);
        if (data.loss >= 0) {
          setLoss(data.loss);
          setPhase("training");
          setLossHistory((prev) => (prev.length && prev[prev.length - 1].step === data.step) ? prev : [...prev.slice(-299), { step: data.step, loss: data.loss }]);
        }
      }
      if (data.type === "done") { setBanner({ type: "success", text: `✓ Saved → ${data.path}` }); setPhase("done"); setFinished(true); setTraining(false); refreshTrained(); }
      if (data.type === "error") { setBanner({ type: "error", text: data.message }); setPhase("error"); setFinished(true); setTraining(false); }
      if (data.type === "status" && data.status === "stopped") { setBanner({ type: "warn", text: "Training stopped." }); setPhase("stopped"); setFinished(true); setTraining(false); }
    };
    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [logs]);

  // ── Actions ──
  function validate(): string | null {
    const model = state.localPath.trim() || state.modelId;
    if (!model) return "Select or enter a base model.";
    if (!state.datasetPath.trim()) return "Enter a dataset path or HuggingFace ID.";
    if (state.numEpochs < 1 || state.batchSize < 1) return "Epochs and batch size must be ≥ 1.";
    return null;
  }
  async function handleStart() {
    const err = validate();
    if (err) { setError(err); return; }
    setError(""); setBanner(null); setFinished(false);
    setLossHistory([]); setLogs([]); setLoss(-1); setStep(0); setTotal(0);
    setPhase("loading"); setTraining(true); setStartTime(Date.now()); setEta(-1); setDownloadProgress(null);
    setTab("train");
    try { const r = await startTraining(state); setJobId(r.id); }
    catch (e) { setTraining(false); setError(e instanceof Error ? e.message : String(e)); }
  }
  async function handleStop() {
    if (finished) { setJobId(""); setFinished(false); setTraining(false); setPhase(""); setBanner(null); return; }
    if (jobId) await stopJob(jobId);
  }
  function applyPreset(i: number) {
    const p = PRESETS[i];
    setState((prev) => ({ ...prev, numEpochs: p.epochs, batchSize: p.batchSize, learningRate: p.learningRate, gradAccum: p.gradAccum, loraR: p.loraR }));
    setActivePreset(i); setError("");
  }
  function selectModel(id: string) {
    // Always select by HF id — transformers loads cached models from the local
    // cache automatically (no re-download). The local-folder field is reserved
    // for genuine on-disk checkpoints the user points at manually.
    setModelQuery(id);
    patch({ modelId: id, localPath: "" });
  }
  function retrain(m: TrainedModel) {
    patch({ mode: m.mode, modelId: m.base || "", localPath: "" });
    setModelQuery(m.base || "");
    setTab("studio");
  }

  // ── Computed ──
  const isActive = training && !finished;
  const pct = total > 0 && step > 0 ? Math.round((step / total) * 100) : 0;
  const method = methodLabel(state);
  const vramEst = estimateVram(method, state.batchSize);
  const totalVram = appConfig?.vram_gb ?? 16;
  const vramPct = Math.min(100, (vramEst / totalVram) * 100);
  const lrIndex = LR_VALUES.findIndex((v) => Math.abs(v - state.learningRate) / v < 0.01);
  const modelStatus: { ok: boolean; text: string } | null = state.localPath.trim()
    ? { ok: true, text: "✓ Using local folder — no download needed" }
    : state.modelId
      ? cachedModels.includes(state.modelId)
        ? { ok: true, text: "✓ Already on disk — no download needed" }
        : { ok: false, text: "↓ Not on disk — will download from HuggingFace when training starts" }
      : null;
  const modelList: ModelInfo[] = state.mode === "cnn" ? popular.cnn : popular.llm;
  const filteredModelList = modelList.filter((m) => !modelQuery || m.id.toLowerCase().includes(modelQuery.toLowerCase()) || m.label.toLowerCase().includes(modelQuery.toLowerCase()));

  const totalSize = trained.reduce((a, m) => a + m.size_bytes, 0);
  const llmCount = trained.filter((m) => m.mode === "llm").length;
  const cnnCount = trained.filter((m) => m.mode === "cnn").length;
  const homeModels = trained.filter((m) => (homeFilter === "all" || m.mode === homeFilter) && (!homeSearch || m.name.toLowerCase().includes(homeSearch.toLowerCase())));

  return (
    <div className="dash">
      {/* ── Topbar ── */}
      <div className="topbar">
        <div className="logo"><span className="logo-mark">✦</span> llmtune <span className="badge">STUDIO</span></div>
        <div className="tr">
          {appConfig && (
            <div className="chip ok"><span className="chip-dot" /> {appConfig.device_label}{appConfig.vram_gb ? ` · ${appConfig.vram_gb} GB` : ""}</div>
          )}
          <div className="chip">🔒 100% local</div>
          {!skipAuth && onLogout && <button className="signout-btn" onClick={onLogout}>Sign out</button>}
        </div>
      </div>

      {/* ── Nav ── */}
      <div className="nav">
        <button className={`ntab ${tab === "home" ? "a" : ""}`} onClick={() => setTab("home")}>🏠 Home</button>
        <button className={`ntab ${tab === "studio" ? "a" : ""}`} onClick={() => setTab("studio")}>🎛 Studio</button>
        <button className={`ntab ${tab === "train" ? "a" : ""}`} onClick={() => setTab("train")}>▶ Train{isActive ? " ●" : ""}</button>
        <button className={`ntab ${tab === "models" ? "a" : ""}`} onClick={() => setTab("models")}>📦 My models</button>
      </div>

      <div className="body">
        {tab === "home" && (
          <div className="tab-panel">
            <div className="home-body">
              <div className="home-left">
                <div className="stat-grid">
                  <div className="scard"><div className="sn">{trained.length}</div><div className="sl">Models trained</div></div>
                  <div className="scard"><div className="sn">{llmCount}</div><div className="sl">LLM models</div></div>
                  <div className="scard"><div className="sn">{cnnCount}</div><div className="sl">CNN models</div></div>
                  <div className="scard"><div className="sn" style={{ fontSize: 16 }}>{formatBytes(totalSize)}</div><div className="sl">Disk used</div></div>
                </div>

                <div className="card">
                  <div className="ch">
                    <div className="ct">📦 Fine-tuned models on this machine</div>
                    <button className="abtn primary" onClick={() => setTab("studio")}>＋ New run</button>
                  </div>
                  <div className="search-bar">
                    <input placeholder="Search models…" value={homeSearch} onChange={(e) => setHomeSearch(e.target.value)} />
                    <div className="filter-pills">
                      {(["all", "llm", "cnn"] as const).map((f) => (
                        <button key={f} className={`fp ${homeFilter === f ? "a" : ""}`} onClick={() => setHomeFilter(f)}>{f === "all" ? "All" : f.toUpperCase()}</button>
                      ))}
                    </div>
                  </div>
                  {homeModels.length === 0 ? (
                    <div className="empty">No fine-tuned models yet.<br />Start a run in the Studio tab to see it here.</div>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {homeModels.map((m) => <ModelCard key={m.path} m={m} onRetrain={() => retrain(m)} hfToken={state.hfToken} />)}
                    </div>
                  )}
                </div>
              </div>

              <div className="home-right">
                <div className="card-r">
                  <div className="ch"><div className="ct">🕘 Recent runs</div></div>
                  {trained.slice(0, 6).map((m) => (
                    <div className="run-row" key={m.path}>
                      <div className="run-dot" style={{ background: "var(--accent)" }} />
                      <div className="run-info">
                        <div className="run-name">{m.name}</div>
                        <div className="run-meta">{m.mode.toUpperCase()} · {formatBytes(m.size_bytes)} · {fmtDate(m.modified)}</div>
                      </div>
                      <span className="pill p-green">done</span>
                    </div>
                  ))}
                  {trained.length === 0 && <div className="hint">No runs recorded yet.</div>}
                </div>

                <div className="card-r">
                  <div className="ch"><div className="ct">⤓ Base models on disk</div></div>
                  {cachedModels.length === 0 ? <div className="hint">No base models cached yet.</div> : cachedModels.slice(0, 8).map((id) => (
                    <div className="run-row" key={id}>
                      <div className="run-info"><div className="run-name">{id}</div></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {tab === "studio" && (
          <div className="tab-panel">
            <div className="two-panel">
              {/* Left: model + dataset */}
              <div className="panel">
                <div className="card">
                  <div className="ch">
                    <div className="ct">🧠 Model</div>
                    <div className="seg" style={{ width: "auto" }}>
                      {(["llm", "cnn"] as const).map((m) => (
                        <button key={m} className={`sb ${state.mode === m ? "a" : ""}`} onClick={() => { patch({ mode: m, modelId: "", localPath: "" }); setModelQuery(""); }}>{m.toUpperCase()}</button>
                      ))}
                    </div>
                  </div>
                  <div className="iw" style={{ marginBottom: 6 }}>
                    <input placeholder="Search or paste model ID…" value={modelQuery} onChange={(e) => { setModelQuery(e.target.value); patch({ modelId: e.target.value, localPath: "" }); }} />
                    <span className="iw-icon">⌕</span>
                  </div>
                  <div className="mlist">
                    {filteredModelList.map((m) => {
                      const cached = cachedModels.includes(m.id);
                      return (
                        <button key={m.id} className={`mr ${state.modelId === m.id ? "sel" : ""}`} onClick={() => selectModel(m.id)}>
                          <div><div className="mn">{m.label}</div><div className="md">{m.size}</div></div>
                          <span className={`pill ${cached ? "p-green" : "p-gray"}`}>{cached ? "✓ on disk" : "download"}</span>
                        </button>
                      );
                    })}
                    {filteredModelList.length === 0 && <div className="hint" style={{ padding: 10 }}>Type a full HuggingFace ID (e.g. Qwen/Qwen2.5-1.5B) to use a custom model.</div>}
                  </div>

                  {/* Local already-downloaded model folder */}
                  <div style={{ marginTop: 10 }}>
                    <div className="lbl">Local model folder <span style={{ textTransform: "none", color: "var(--color-text-tertiary)", fontWeight: 400 }}>· optional, point to a model you already downloaded</span></div>
                    <div className="iw">
                      <input placeholder="~/models/my-model  (leave empty to download from HuggingFace)" value={state.localPath} onChange={(e) => patch({ localPath: e.target.value })} />
                      <span className="iw-icon">📂</span>
                    </div>
                    {modelStatus && <div className={modelStatus.ok ? "ds-ok" : "hint"} style={{ marginTop: 6 }}>{modelStatus.text}</div>}
                  </div>

                  <div className="g2" style={{ marginTop: 10 }}>
                    <div>
                      <div className="lbl">Method</div>
                      <div className="seg">
                        {state.mode === "llm" ? (
                          <>
                            <button className={`sb ${state.quantization === "none" && state.trainingMode !== "full" ? "a" : ""}`} onClick={() => patch({ quantization: "none", trainingMode: "lora" })}>LoRA</button>
                            <button className={`sb ${state.quantization === "4bit" ? "a" : ""}`} onClick={() => patch({ quantization: "4bit", trainingMode: "lora" })}>QLoRA</button>
                            <button className={`sb ${state.trainingMode === "full" ? "a" : ""}`} onClick={() => patch({ trainingMode: "full", quantization: "none" })}>Full</button>
                          </>
                        ) : (
                          <>
                            <button className={`sb ${state.trainingMode === "feature_extraction" ? "a" : ""}`} onClick={() => patch({ trainingMode: "feature_extraction" })}>Head</button>
                            <button className={`sb ${state.trainingMode === "lora" ? "a" : ""}`} onClick={() => patch({ trainingMode: "lora" })}>LoRA</button>
                            <button className={`sb ${state.trainingMode === "full" ? "a" : ""}`} onClick={() => patch({ trainingMode: "full" })}>Full</button>
                          </>
                        )}
                      </div>
                    </div>
                    <div>
                      <div className="lbl">HF token</div>
                      <div className="iw"><input type="password" placeholder="hf_…" value={state.hfToken} onChange={(e) => patch({ hfToken: e.target.value })} /><span className="iw-icon">🔑</span></div>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="ch">
                    <div className="ct">🗄 Dataset</div>
                    {state.mode === "llm" && (
                      <div className="seg" style={{ width: "auto" }}>
                        <button className={`sb ${dsSource === "local" ? "a" : ""}`} onClick={() => setDsSource("local")}>Local / HF</button>
                        <button className={`sb ${dsSource === "kaggle" ? "a" : ""}`} onClick={() => setDsSource("kaggle")}>Kaggle</button>
                      </div>
                    )}
                  </div>

                  {dsSource === "kaggle" && state.mode === "llm" && (
                    <KagglePanel
                      onPick={(p) => { patch({ datasetPath: p, datasetFormat: p.endsWith(".csv") ? "csv" : p.endsWith(".json") ? "json" : p.endsWith(".txt") ? "text" : "jsonl" }); setDsSource("local"); }}
                      onUseHf={(id) => { patch({ datasetPath: id }); setDsSource("local"); }}
                    />
                  )}

                  {(dsSource === "local" || state.mode === "cnn") && (<>
                  <div className="iw" style={{ marginBottom: 8 }}>
                    <input placeholder={state.mode === "cnn" ? "/path/to/class-folders" : "/path/to/data.jsonl  or  owner/dataset"} value={state.datasetPath} onChange={(e) => patch({ datasetPath: e.target.value })} />
                    <span className="iw-icon">📁</span>
                  </div>
                  {state.mode === "llm" && (
                    <div className="g2">
                      <div>
                        <div className="lbl">Format</div>
                        <div className="seg">
                          {(["jsonl", "json", "csv"] as const).map((f) => (
                            <button key={f} className={`sb ${state.datasetFormat === f ? "a" : ""}`} onClick={() => patch({ datasetFormat: f })}>{f.toUpperCase()}</button>
                          ))}
                        </div>
                      </div>
                      <div>
                        <div className="lbl">Validation</div>
                        <div className="seg">
                          {[{ l: "none", v: 0 }, { l: "10%", v: 0.1 }, { l: "20%", v: 0.2 }].map((o) => (
                            <button key={o.l} className={`sb ${state.validationSplit === o.v ? "a" : ""}`} onClick={() => patch({ validationSplit: o.v })}>{o.l}</button>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                  {datasetStatus?.status === "ok" && (
                    <div className="ds-ok" style={{ marginTop: 8 }}>✓ {datasetStatus.row_count?.toLocaleString()} rows · {datasetStatus.format}{datasetStatus.avg_tokens ? ` · avg ${datasetStatus.avg_tokens} tokens` : ""}</div>
                  )}
                  {datasetStatus && datasetStatus.status !== "ok" && datasetStatus.message && (
                    <div className="ds-err" style={{ marginTop: 8 }}>✗ {datasetStatus.message}</div>
                  )}
                  {isHfId(state.datasetPath) && <div className="hint" style={{ marginTop: 6 }}>🤗 Will download from the HuggingFace Hub.</div>}
                  </>)}
                </div>
              </div>

              {/* Right: preset + params + toggles + start */}
              <div className="panel">
                <div className="card">
                  <div className="ch"><div className="ct">★ Preset</div></div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 5, marginBottom: 12 }}>
                    {PRESETS.map((p, i) => (
                      <button key={p.label} className={`pcard ${activePreset === i ? "a" : ""}`} onClick={() => applyPreset(i)}>
                        <div style={{ fontSize: 14 }}>{p.icon}</div>
                        <div style={{ fontSize: 10, fontWeight: 600, marginTop: 2 }}>{p.label}</div>
                      </button>
                    ))}
                  </div>
                  <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                    <span>VRAM estimate</span><span style={{ color: "var(--accent)", fontWeight: 600 }}>~{vramEst.toFixed(1)} GB of {totalVram} GB</span>
                  </div>
                  <div className="vb"><div className="vbf" style={{ width: `${vramPct}%` }} /></div>
                  <div style={{ height: 8 }} />
                  <Slider label="Epochs" min={1} max={20} value={state.numEpochs} onChange={(v) => { patch({ numEpochs: v }); setActivePreset(null); }} />
                  <Slider label="Batch size" min={1} max={16} value={state.batchSize} onChange={(v) => { patch({ batchSize: v }); setActivePreset(null); }} />
                  <div className="pr">
                    <span className="prl">Learning rate</span>
                    <input type="range" min={0} max={LR_VALUES.length - 1} step={1} value={lrIndex >= 0 ? lrIndex : 4} onChange={(e) => { patch({ learningRate: LR_VALUES[parseInt(e.target.value)] }); setActivePreset(null); }} />
                    <span className="prv">{formatLR(state.learningRate)}</span>
                  </div>
                  {state.mode === "llm" && (
                    <>
                      <Slider label="Grad accum" min={1} max={32} value={state.gradAccum} onChange={(v) => { patch({ gradAccum: v }); setActivePreset(null); }} />
                      <Slider label="Warmup steps" min={0} max={200} step={10} value={state.warmupSteps} onChange={(v) => patch({ warmupSteps: v })} />
                    </>
                  )}
                </div>

                <div className="card">
                  <Toggle label="Save checkpoints" value={state.saveCheckpoints} onChange={(v) => patch({ saveCheckpoints: v })} />
                  <Toggle label="Resume from last checkpoint" value={state.resumeFromCheckpoint} onChange={(v) => patch({ resumeFromCheckpoint: v })} />
                  {state.mode === "llm" && <Toggle label="Flash attention" value={state.flashAttention} onChange={(v) => patch({ flashAttention: v })} />}
                  <Toggle label="Push to HF Hub after training" value={state.pushToHub} onChange={(v) => patch({ pushToHub: v })} last={!state.pushToHub} />
                  {state.pushToHub && (
                    <div style={{ marginTop: 8 }}>
                      <div className="iw"><input placeholder="username/model-name" value={state.hubModelId} onChange={(e) => patch({ hubModelId: e.target.value })} /><span className="iw-icon">🤗</span></div>
                      {!state.hfToken && <div className="hint" style={{ marginTop: 4 }}>Add a HF token (in Model panel) with write access to push.</div>}
                    </div>
                  )}
                </div>

                <div className="card">
                  <div className="lbl" style={{ marginBottom: 4 }}>Output folder</div>
                  <div className="iw" style={{ marginBottom: 10 }}>
                    <input value={state.outputDir} placeholder={appConfig?.default_llm_output ?? "~/llmtune-output"} onChange={(e) => patch({ outputDir: e.target.value })} />
                    <span className="iw-icon">📂</span>
                  </div>
                  {error && <p className="form-error" style={{ marginBottom: 8 }}>{error}</p>}
                  <button className="startbtn" onClick={handleStart} disabled={isActive}>▶ Start training</button>
                </div>
              </div>
            </div>
          </div>
        )}

        {tab === "train" && (
          <div className="tab-panel">
            <div className="two-panel">
              <div className="panel">
                <div className="card">
                  <div className="ch"><div className="ct">📈 Training</div>{isActive && <span className="live-tag">{connected ? "● Live" : "Polling"}</span>}</div>
                  <div className="stat3">
                    <div className="stat"><div className="stat-n">{total > 0 ? `${step}/${total}` : "—"}</div><div className="stat-l">Step</div></div>
                    <div className="stat"><div className="stat-n">{loss >= 0 ? loss.toFixed(3) : "—"}</div><div className="stat-l">Loss</div></div>
                    <div className="stat"><div className="stat-n">{fmtEta(eta)}</div><div className="stat-l">ETA</div></div>
                  </div>
                  {downloadProgress !== null && downloadProgress < 100 ? (
                    <div style={{ background: "var(--color-background-secondary)", borderRadius: 8, height: 120, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, padding: 14 }}>
                      <div style={{ fontSize: 13, fontWeight: 600 }}>⤓ Downloading model… {downloadProgress}%</div>
                      <div style={{ width: "100%" }} className="vb"><div className="vbf" style={{ width: `${downloadProgress}%` }} /></div>
                      <div style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>{logs.slice().reverse().find((l) => l.startsWith("Downloading:")) || "First run downloads the base model — this is one-time."}</div>
                    </div>
                  ) : lossHistory.length > 0 ? (
                    <LossChart data={lossHistory} height={120} />
                  ) : (
                    <div style={{ background: "var(--color-background-secondary)", borderRadius: 8, height: 120, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 6, color: "var(--color-text-tertiary)", fontSize: 12 }}>
                      {isActive && <span className="spinner" style={{ width: 18, height: 18, border: "2px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", display: "inline-block", animation: "spin 0.8s linear infinite" }} />}
                      <span>{isActive ? `${PHASE_LABELS[phase] ?? phase}…` : "Loss chart appears once training starts"}</span>
                    </div>
                  )}
                  <div className="vb" style={{ margin: "8px 0" }}><div className="vbf" style={{ width: `${downloadProgress !== null && downloadProgress < 100 ? downloadProgress : Math.max(pct, isActive ? 2 : 0)}%` }} /></div>
                  {banner && <div className={`toast ${banner.type}`} style={{ marginBottom: 8 }}>{banner.text}</div>}
                  <button className={`startbtn ${isActive ? "stop" : finished ? "" : "ghost"}`} onClick={isActive || finished ? handleStop : () => setTab("studio")}>
                    {finished ? "↻ New run" : isActive ? "■ Stop training" : "← Back to studio"}
                  </button>
                </div>
                <div className="card" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
                  <div className="ch"><div className="ct">⌗ Log</div><button className="abtn" onClick={() => navigator.clipboard?.writeText(logs.join("\n"))}>Copy</button></div>
                  <div className="logbox" ref={logRef}>{logs.length ? logs.slice(-200).join("\n") : "Waiting for training to start…"}</div>
                </div>
              </div>
              <div className="panel">
                <div className="card">
                  <div className="ct" style={{ marginBottom: 8 }}>ⓘ Run config</div>
                  <div className="kv"><span>Model</span><span>{(state.localPath.trim() || state.modelId) || "—"}</span></div>
                  <div className="kv"><span>Method</span><span>{method}</span></div>
                  <div className="kv"><span>Dataset</span><span>{state.datasetPath ? state.datasetPath.split("/").pop() : "—"}</span></div>
                  <div className="kv"><span>Epochs / LR</span><span>{state.numEpochs} · {formatLR(state.learningRate)}</span></div>
                  <div className="kv"><span>Output</span><span>{state.outputDir ? state.outputDir.split("/").pop() : "—"}</span></div>
                </div>
                <div className="card">
                  <div className="ct" style={{ marginBottom: 6 }}>⚙ Device</div>
                  {appConfig && <div className="ds-ok">✓ {appConfig.device.toUpperCase()} · {appConfig.device_label}{appConfig.vram_gb ? ` · ${appConfig.vram_gb} GB` : ""}</div>}
                  {downloadProgress !== null && downloadProgress < 100 && (
                    <><div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 8 }}>Downloading model… {downloadProgress}%</div><div className="vb"><div className="vbf" style={{ width: `${downloadProgress}%` }} /></div></>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {tab === "models" && (
          <div className="tab-panel">
            <div style={{ padding: 14, overflowY: "auto", display: "flex", flexDirection: "column", gap: 10, flex: 1 }}>
              <div className="card">
                <div className="ch">
                  <div className="ct">📦 My fine-tuned models</div>
                  <button className="abtn primary" onClick={() => setTab("studio")}>＋ New run</button>
                </div>
                {trained.length === 0 ? (
                  <div className="empty">No fine-tuned models on this machine yet.<br />Train one in the Studio tab — it will appear here automatically.</div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {trained.map((m) => <ModelCard key={m.path} m={m} onRetrain={() => retrain(m)} hfToken={state.hfToken} full />)}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ──
function ModelCard({ m, onRetrain, hfToken, full }: { m: TrainedModel; onRetrain: () => void; hfToken?: string; full?: boolean }) {
  const [panel, setPanel] = useState<"" | "test" | "push" | "export">("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  // test
  const [prompt, setPrompt] = useState("");
  const [answer, setAnswer] = useState("");
  // push
  const [repoId, setRepoId] = useState("");
  const [pushUrl, setPushUrl] = useState("");
  // export
  const [quant, setQuant] = useState("q8_0");
  const [exportMsg, setExportMsg] = useState("");

  const isLlm = m.mode === "llm";

  function toggle(p: "test" | "push" | "export") { setErr(""); setPanel((cur) => (cur === p ? "" : p)); }

  async function runTest() {
    setErr(""); setAnswer(""); setBusy(true);
    try { setAnswer(await inferModel(m.path, prompt, hfToken)); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  async function runPush() {
    setErr(""); setPushUrl(""); setBusy(true);
    try {
      if (!hfToken) throw new Error("Add a HuggingFace token in the Studio Model panel first (needs write access).");
      setPushUrl(await pushModel(m.path, repoId.trim(), hfToken, true));
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  async function runExport() {
    setErr(""); setExportMsg(""); setBusy(true);
    try { const r = await exportGguf(m.path, quant, hfToken); setExportMsg(r.message); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  return (
    <div className="model-card">
      <div className="mc-top">
        <div className="mc-name">{m.mode === "cnn" ? "🖼" : "✦"} {m.name}</div>
        <div className="mc-base">Base: {m.base || "unknown"} · {fmtDate(m.modified)}</div>
        <div className="mc-meta">
          <span className="pill p-blue">{m.mode.toUpperCase()}</span>
          <span className="pill p-gray">{formatBytes(m.size_bytes)}</span>
        </div>
      </div>
      <div className="mc-bot">
        <span className="mc-path">{m.path}</span>
        <div className="mc-actions">
          {full && isLlm && <button className={`abtn ${panel === "test" ? "primary" : ""}`} onClick={() => toggle("test")}>💬 Test</button>}
          {full && <button className={`abtn ${panel === "push" ? "primary" : ""}`} onClick={() => toggle("push")}>⤴ Push</button>}
          {full && isLlm && <button className={`abtn ${panel === "export" ? "primary" : ""}`} onClick={() => toggle("export")}>⇩ GGUF</button>}
          <button className="abtn" onClick={() => navigator.clipboard?.writeText(m.path)}>Copy path</button>
          <button className="abtn" onClick={onRetrain}>↻ Retrain</button>
        </div>
      </div>

      {panel && (
        <div style={{ padding: "10px 14px", borderTop: "1px solid var(--color-border-tertiary)", display: "flex", flexDirection: "column", gap: 8 }}>
          {panel === "test" && (<>
            <textarea className="logbox" style={{ minHeight: 56, fontFamily: "inherit", fontSize: 12 }} placeholder="Ask your fine-tuned model something…" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            <button className="abtn primary" style={{ alignSelf: "flex-start" }} disabled={busy || !prompt.trim()} onClick={runTest}>{busy ? "Generating…" : "Run"}</button>
            {answer && <div className="logbox" style={{ minHeight: 40 }}>{answer}</div>}
          </>)}
          {panel === "push" && (<>
            <div className="iw"><input placeholder="username/model-name" value={repoId} onChange={(e) => setRepoId(e.target.value)} /><span className="iw-icon">🤗</span></div>
            <button className="abtn primary" style={{ alignSelf: "flex-start" }} disabled={busy || !repoId.includes("/")} onClick={runPush}>{busy ? "Uploading…" : "Push to Hub"}</button>
            {pushUrl && <div className="ds-ok">✓ Uploaded — {pushUrl}</div>}
          </>)}
          {panel === "export" && (<>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span className="lbl" style={{ margin: 0 }}>Quant</span>
              <div className="seg" style={{ width: "auto" }}>
                {["q8_0", "q4_k_m", "f16"].map((q) => (
                  <button key={q} className={`sb ${quant === q ? "a" : ""}`} onClick={() => setQuant(q)}>{q}</button>
                ))}
              </div>
              <button className="abtn primary" disabled={busy} onClick={runExport}>{busy ? "Exporting…" : "Export GGUF"}</button>
            </div>
            {busy && <div className="hint">Merging the model and (first time only) setting up llama.cpp — this can take a few minutes.</div>}
            {exportMsg && <div className="hint" style={{ whiteSpace: "pre-wrap" }}>{exportMsg}</div>}
          </>)}
          {err && <div className="ds-err">✗ {err}</div>}
        </div>
      )}
    </div>
  );
}

function Slider({ label, min, max, step = 1, value, onChange }: { label: string; min: number; max: number; step?: number; value: number; onChange: (v: number) => void }) {
  return (
    <div className="pr">
      <span className="prl">{label}</span>
      <input type="range" min={min} max={max} step={step} value={Math.min(value, max)} onChange={(e) => onChange(parseFloat(e.target.value))} />
      <span className="prv">{value}</span>
    </div>
  );
}

function Toggle({ label, value, onChange, last }: { label: string; value: boolean; onChange: (v: boolean) => void; last?: boolean }) {
  return (
    <div className="trow" style={last ? { borderBottom: "none" } : undefined}>
      <span className="tlbl">{label}</span>
      <button className={`tog ${value ? "on" : "off"}`} role="switch" aria-checked={value} onClick={() => onChange(!value)} />
    </div>
  );
}
