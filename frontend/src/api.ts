import type { ModelInfo, TrainMode, WizardState } from "./types";

// ── Base URL ──────────────────────────────────────────────────────────────────
function getBase(): string {
  if (typeof window !== "undefined" && (window as any).__LLMTUNE_API_BASE__)
    return (window as any).__LLMTUNE_API_BASE__;
  if (typeof window !== "undefined" && window.location.origin.startsWith("http"))
    return window.location.origin;
  return "http://127.0.0.1:8765";
}
export const API_BASE = getBase();

// ── Config ────────────────────────────────────────────────────────────────────
export interface AppConfig {
  skip_auth: boolean;
  platform: string;
  device: string;
  device_label: string;
  default_llm_output: string;
  default_cnn_output: string;
  home_dir: string;
  vram_gb: number | null;
}

export async function fetchConfig(): Promise<AppConfig> {
  const res = await fetch(`${API_BASE}/api/config`);
  return res.json();
}

// ── Models ────────────────────────────────────────────────────────────────────
export async function fetchModels(): Promise<{ llm: ModelInfo[]; cnn: ModelInfo[] }> {
  const res = await fetch(`${API_BASE}/api/models`);
  return res.json();
}

export async function searchHFModels(
  q: string,
  task = "text-generation"
): Promise<{ id: string; label: string; downloads: number; size: string | null }[]> {
  try {
    const res = await fetch(
      `${API_BASE}/api/models/search?q=${encodeURIComponent(q)}&task=${encodeURIComponent(task)}`
    );
    return res.ok ? res.json() : [];
  } catch {
    return [];
  }
}

export interface TrainedModel {
  name: string;
  path: string;
  mode: "llm" | "cnn";
  base: string;
  size_bytes: number;
  modified: number;
}

export async function fetchTrainedModels(): Promise<TrainedModel[]> {
  try {
    const res = await fetch(`${API_BASE}/api/models/trained`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.models || [];
  } catch {
    return [];
  }
}

export async function fetchCachedModels(): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/api/models/cached`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.models || [];
  } catch {
    return [];
  }
}

// ── Training ──────────────────────────────────────────────────────────────────
export async function startTraining(state: WizardState): Promise<{ id: string }> {
  const modelId = state.localPath.trim() || state.modelId;
  const outputDir = state.outputDir || undefined;

  const body =
    state.mode === "cnn"
      ? {
          mode: "cnn" as const,
          model_id: modelId,
          hf_token: state.hfToken || null,
          dataset_path: state.datasetPath,
          training_mode: state.trainingMode,
          num_epochs: state.numEpochs,
          per_device_train_batch_size: state.batchSize,
          learning_rate: state.learningRate,
          lora_r: state.loraR,
          lora_alpha: state.loraAlpha,
          lora_dropout: state.loraDropout,
          output_dir: outputDir,
          save_checkpoints: state.saveCheckpoints,
        }
      : {
          mode: "llm" as const,
          model_id: modelId,
          hf_token: state.hfToken || null,
          dataset_path: state.datasetPath,
          dataset_format: state.datasetFormat,
          text_column: state.textColumn,
          max_seq_length: state.maxSeqLength,
          num_epochs: state.numEpochs,
          per_device_train_batch_size: state.batchSize,
          gradient_accumulation_steps: state.gradAccum,
          learning_rate: state.learningRate,
          warmup_ratio: state.warmupRatio,
          warmup_steps: state.warmupSteps,
          lora_r: state.loraR,
          lora_alpha: state.loraAlpha,
          lora_dropout: state.loraDropout,
          target_modules: state.targetModules,
          quantization: state.quantization,
          lr_scheduler_type: state.lrSchedulerType,
          validation_split: state.validationSplit,
          pack_sequences: state.packSequences,
          shuffle: state.shuffleDataset,
          output_dir: outputDir,
          save_checkpoints: state.saveCheckpoints,
          eval_on_validation: state.evalOnValidation,
          resume_from_checkpoint: state.resumeFromCheckpoint,
          push_to_hub: state.pushToHub,
          hub_model_id: state.hubModelId || null,
        };

  const res = await fetch(`${API_BASE}/api/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data?.detail ?? `Failed to start training (${res.status})`);
  }
  return res.json();
}

// ── Model library actions ─────────────────────────────────────────────────────
export async function inferModel(path: string, prompt: string, hfToken?: string): Promise<string> {
  const res = await fetch(`${API_BASE}/api/library/infer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, prompt, hf_token: hfToken || null }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data?.detail ?? `Inference failed (${res.status})`);
  }
  return (await res.json()).output as string;
}

export async function pushModel(path: string, repoId: string, token: string, isPrivate = true): Promise<string> {
  const res = await fetch(`${API_BASE}/api/library/push`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, repo_id: repoId, token, private: isPrivate }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data?.detail ?? `Push failed (${res.status})`);
  }
  return (await res.json()).url as string;
}

export interface GgufResult {
  status: "ok" | "merged_only";
  gguf: string;
  merged: string;
  message: string;
}
export async function exportGguf(path: string, quant: string, hfToken?: string): Promise<GgufResult> {
  const res = await fetch(`${API_BASE}/api/library/export-gguf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, quant, hf_token: hfToken || null }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data?.detail ?? `Export failed (${res.status})`);
  }
  return res.json();
}

export async function stopJob(jobId: string): Promise<void> {
  await fetch(`${API_BASE}/api/jobs/${jobId}/stop`, { method: "POST" });
}

export interface JobStatus {
  id: string;
  mode: string;
  status: string;
  step: number;
  total_steps: number;
  loss: number;
  phase: string;
  device: string;
  output_path: string;
  error: string;
  logs: string[];
}

export async function pollJob(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error("Job not found");
  return res.json();
}

export function jobWebSocket(jobId: string): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = API_BASE ? new URL(API_BASE).host : window.location.host;
  return new WebSocket(`${proto}//${host}/api/jobs/${jobId}/ws`);
}

export function modeLabel(mode: TrainMode): string {
  return mode === "cnn" ? "Image classification" : "LLM / Text";
}

// ── Kaggle ────────────────────────────────────────────────────────────────────
export interface KaggleDataset {
  ref: string;
  title: string;
  subtitle: string;
  size_bytes: number;
  downloads: number;
  updated: string;
  url: string;
}
export interface KaggleFile {
  path: string;
  name: string;
  ext: string;
  size_bytes: number;
}
export interface KaggleCreds {
  username: string;
  key: string;
}

export async function kaggleSearch(query: string, creds: KaggleCreds): Promise<KaggleDataset[]> {
  const res = await fetch(`${API_BASE}/api/kaggle/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, ...creds }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data?.detail ?? `Kaggle search failed (${res.status})`);
  }
  return (await res.json()).datasets || [];
}

export async function kaggleDownload(ref: string, creds: KaggleCreds): Promise<{ folder: string; files: KaggleFile[] }> {
  const res = await fetch(`${API_BASE}/api/kaggle/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ref, ...creds }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data?.detail ?? `Kaggle download failed (${res.status})`);
  }
  return res.json();
}

export interface DatasetValidation {
  status: "ok" | "empty" | "not_found" | "not_file" | "parse_error" | "error";
  message?: string;
  row_count?: number;
  columns?: string[];
  format?: string;
  preview?: Record<string, unknown>[];
  avg_tokens?: number;
}

export async function validateDataset(path: string): Promise<DatasetValidation> {
  try {
    const res = await fetch(`${API_BASE}/api/dataset/validate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    return res.ok ? res.json() : { status: "error", message: "Server error" };
  } catch {
    return { status: "error", message: "Network error" };
  }
}
