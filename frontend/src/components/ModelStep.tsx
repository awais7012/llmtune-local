import { useEffect, useState } from "react";
import { fetchModels } from "../api";
import type { ModelInfo, TrainMode, WizardState } from "../types";

interface Props {
  state: WizardState;
  onChange: (patch: Partial<WizardState>) => void;
  onNext: () => void;
}

export default function ModelStep({ state, onChange, onNext }: Props) {
  const [llmModels, setLlmModels] = useState<ModelInfo[]>([]);
  const [cnnModels, setCnnModels] = useState<ModelInfo[]>([]);
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    fetchModels().then((data) => {
      setLlmModels(data.llm);
      setCnnModels(data.cnn);
    });
  }, []);

  const models = state.mode === "cnn" ? cnnModels : llmModels;
  const filtered = models.filter(
    (m) =>
      !search ||
      m.id.toLowerCase().includes(search.toLowerCase()) ||
      m.label.toLowerCase().includes(search.toLowerCase())
  );

  const selectedId = state.localPath.trim() || state.modelId;

  function selectModel(id: string, mode: TrainMode) {
    onChange({ modelId: id, localPath: "", mode });
    setError("");
  }

  function handleNext() {
    const id = state.localPath.trim() || state.modelId;
    if (!id) {
      setError("Pick a model or enter a local path to continue.");
      return;
    }
    setError("");
    onNext();
  }

  return (
    <div>
      <h2 className="screen-title">Select Base Model</h2>
      <p className="screen-subtitle">Choose a model to fine-tune on your machine.</p>

      {selectedId && (
        <div className="selected-model">
          Selected: <strong>{selectedId}</strong>
          {state.mode === "cnn" && " · Image model"}
        </div>
      )}

      <div className="field">
        <label>Local model folder</label>
        <p className="hint">Already downloaded? Paste the path to your HuggingFace model folder.</p>
        <input
          type="text"
          placeholder="/Users/you/models/llama3"
          value={state.localPath}
          onChange={(e) => {
            onChange({ localPath: e.target.value, modelId: "" });
            setError("");
          }}
        />
      </div>

      <div className="tabs">
        <button
          className={`tab ${state.mode === "llm" ? "active" : ""}`}
          onClick={() => onChange({ mode: "llm", modelId: "" })}
        >
          LLM / Text
        </button>
        <button
          className={`tab ${state.mode === "cnn" ? "active" : ""}`}
          onClick={() => onChange({ mode: "cnn", modelId: "" })}
        >
          CNN / Image
        </button>
      </div>

      <div className="field">
        <input
          type="text"
          placeholder={`Search ${state.mode === "cnn" ? "CNN" : "LLM"} models…`}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="model-list">
        {filtered.map((m) => (
          <button
            key={m.id}
            className={`model-item ${state.modelId === m.id ? "selected" : ""}`}
            onClick={() => selectModel(m.id, state.mode)}
          >
            <span className="name">{m.label}</span>
            <span className="size">{m.size}</span>
          </button>
        ))}
      </div>

      <details className="collapsible">
        <summary>HuggingFace Token — for gated / private models</summary>
        <div className="content">
          <p className="hint">
            Required for gated models (Llama, Gemma). Get a free token at huggingface.co/settings/tokens
          </p>
          <input
            type="password"
            placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
            value={state.hfToken}
            onChange={(e) => onChange({ hfToken: e.target.value })}
          />
        </div>
      </details>

      <div className="bottom-nav">
        <span className="status-msg">{error}</span>
        <button className="btn btn-primary" onClick={handleNext}>
          Next →
        </button>
      </div>
    </div>
  );
}
