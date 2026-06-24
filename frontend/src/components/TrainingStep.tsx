import { useEffect, useRef, useState } from "react";
import { jobWebSocket, pollJob, stopJob } from "../api";
import type { WizardState } from "../types";

interface Props {
  state: WizardState;
  jobId: string;
  onRestart: () => void;
}

const PHASE_LABELS: Record<string, string> = {
  loading: "Loading model",
  downloading: "Downloading weights",
  dataset: "Preparing dataset",
  training: "Training",
  done: "Complete",
  error: "Failed",
  stopped: "Stopped",
};

const DEVICE_LABELS: Record<string, string> = {
  MPS: "Apple Silicon GPU",
  CUDA: "NVIDIA GPU (CUDA)",
  CPU: "CPU",
};

function lossClass(loss: number): string {
  if (loss <= 0) return "";
  if (loss < 0.5) return "loss-low";
  if (loss < 1.5) return "loss-mid";
  return "loss-high";
}

export default function TrainingStep({ state, jobId, onRestart }: Props) {
  const [phase, setPhase] = useState("loading");
  const [device, setDevice] = useState("Detecting…");
  const [step, setStep] = useState(0);
  const [total, setTotal] = useState(0);
  const [loss, setLoss] = useState(-1);
  const [logs, setLogs] = useState<string[]>([]);
  const [banner, setBanner] = useState<{ type: "success" | "error" | "warn"; text: string } | null>(null);
  const [finished, setFinished] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [connected, setConnected] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const modelName = (state.localPath || state.modelId).split("/").pop() ?? "model";
  const pct = total > 0 && loss >= 0 ? Math.round((step / total) * 100) : loss < 0 && total > 0 ? step : 0;
  const isLoading = !finished && ["loading", "downloading", "dataset"].includes(phase);
  const lastLog = logs[logs.length - 1] ?? "Starting…";

  function applyJob(data: {
    status?: string;
    step?: number;
    total_steps?: number;
    loss?: number;
    phase?: string;
    device?: string;
    logs?: string[];
    output_path?: string;
    error?: string;
  }) {
    if (data.phase) setPhase(data.phase);
    if (data.device) setDevice(DEVICE_LABELS[data.device] ?? data.device);
    if (data.step !== undefined) setStep(data.step);
    if (data.total_steps !== undefined) setTotal(data.total_steps);
    if (data.loss !== undefined) setLoss(data.loss);
    if (data.logs?.length) setLogs(data.logs);

    if (data.status === "completed" || data.output_path) {
      setBanner({ type: "success", text: `Saved → ${data.output_path}` });
      setPhase("done");
      setFinished(true);
    } else if (data.status === "failed" || data.error) {
      setBanner({ type: "error", text: data.error || "Training failed" });
      setPhase("error");
      setFinished(true);
    } else if (data.status === "stopped") {
      setBanner({ type: "warn", text: "Training stopped." });
      setPhase("stopped");
      setFinished(true);
    }
  }

  // Polling fallback — keeps UI alive even if WebSocket drops
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      if (!alive || finished) return;
      try {
        const data = await pollJob(jobId);
        applyJob(data);
      } catch { /* ignore */ }
    };
    poll();
    const id = setInterval(poll, 1500);
    return () => { alive = false; clearInterval(id); };
  }, [jobId, finished]);

  // WebSocket for real-time updates
  useEffect(() => {
    const ws = jobWebSocket(jobId);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.type === "snapshot") {
        applyJob({
          status: data.status,
          step: data.step,
          total_steps: data.total,
          loss: data.loss,
          phase: data.phase,
          device: data.device,
          logs: data.logs,
          output_path: data.output_path,
          error: data.error,
        });
      }
      if (data.type === "log") {
        setLogs((prev) => [...prev.slice(-499), data.message]);
        const lower = data.message.toLowerCase();
        if (lower.includes("preprocessing") || lower.includes("scanning") || lower.includes("loading dataset")) {
          setPhase("dataset");
        } else if (lower.includes("starting training")) {
          setPhase("training");
        }
      }
      if (data.type === "progress") {
        setStep(data.step);
        setTotal(data.total);
        setLoss(data.loss);
        if (data.loss >= 0) setPhase("training");
      }
      if (data.type === "done") {
        setBanner({ type: "success", text: `Saved → ${data.path}` });
        setPhase("done");
        setFinished(true);
      }
      if (data.type === "error") {
        setBanner({ type: "error", text: data.message });
        setPhase("error");
        setFinished(true);
      }
      if (data.type === "status" && data.status === "stopped") {
        setBanner({ type: "warn", text: "Training stopped." });
        setPhase("stopped");
        setFinished(true);
      }
    };

    return () => { ws.close(); wsRef.current = null; };
  }, [jobId]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  async function handleStop() {
    if (finished) { onRestart(); return; }
    setStopping(true);
    await stopJob(jobId);
    setStopping(false);
  }

  return (
    <div className="training-screen">
      <div className={`status-hero ${isLoading ? "pulsing" : ""} ${finished ? "done" : ""}`}>
        <div className="status-hero-top">
          <span className="status-phase">{PHASE_LABELS[phase] ?? phase}</span>
          <span className={`conn-dot ${connected ? "on" : "off"}`} title={connected ? "Live" : "Polling"} />
        </div>
        <p className="status-model">{modelName}</p>
        <p className="status-sub">{device}</p>
        {isLoading && (
          <p className="status-hint">
            {lastLog}
            <span className="blink"> …</span>
          </p>
        )}
      </div>

      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-label">Step</div>
          <div className="metric-value">{loss >= 0 && total > 0 ? `${step} / ${total}` : "—"}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Loss</div>
          <div className={`metric-value ${lossClass(loss)}`}>{loss >= 0 ? loss.toFixed(4) : "—"}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Progress</div>
          <div className="metric-value">{pct > 0 ? `${pct}%` : isLoading ? "…" : "—"}</div>
        </div>
      </div>

      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${Math.max(pct, isLoading ? 3 : 0)}%` }} />
      </div>

      <div className="log-panel" ref={logRef}>
        {logs.length === 0 ? (
          <div className="log-line dim">Waiting for logs…</div>
        ) : (
          logs.map((line, i) => <div key={i} className="log-line">{line}</div>)
        )}
      </div>

      {banner && <div className={`banner ${banner.type}`}>{banner.text}</div>}

      <div className="bottom-nav">
        <span className="hint">{isLoading ? "Dataset prep can take several minutes on large folders." : ""}</span>
        <button
          className={`btn ${finished ? "btn-primary" : "btn-danger"}`}
          onClick={handleStop}
          disabled={stopping}
        >
          {finished ? "New Run" : stopping ? "Stopping…" : "Stop"}
        </button>
      </div>
    </div>
  );
}
