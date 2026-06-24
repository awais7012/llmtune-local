import { useState } from "react";
import type { WizardState } from "../types";

interface Props {
  state: WizardState;
  onChange: (patch: Partial<WizardState>) => void;
  onNext: () => void;
  onBack: () => void;
}

function isHfDatasetId(path: string): boolean {
  const p = path.trim();
  if (!p || p.startsWith("/") || p.startsWith("~") || p.startsWith(".")) return false;
  const parts = p.split("/");
  return parts.length === 2 && parts.every(Boolean);
}

export default function DatasetStep({ state, onChange, onNext, onBack }: Props) {
  const [error, setError] = useState("");

  const showHfToken = isHfDatasetId(state.datasetPath);

  function handleNext() {
    if (!state.datasetPath.trim()) {
      setError("Enter a file path or HuggingFace dataset ID.");
      return;
    }
    setError("");
    onNext();
  }

  if (state.mode === "cnn") {
    return (
      <div>
        <h2 className="screen-title">Configure Image Dataset</h2>
        <p className="screen-subtitle">
          Point to a folder with subfolders per class (e.g. cats/, dogs/).
        </p>

        <div className="field">
          <label>Dataset folder</label>
          <input
            type="text"
            placeholder="/Users/you/data/my-images"
            value={state.datasetPath}
            onChange={(e) => {
              onChange({ datasetPath: e.target.value });
              setError("");
            }}
          />
        </div>

        <div className="bottom-nav">
          <button className="btn" onClick={onBack}>← Back</button>
          <span className="status-msg">{error}</span>
          <button className="btn btn-primary" onClick={handleNext}>Next →</button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h2 className="screen-title">Configure Dataset</h2>
      <p className="screen-subtitle">
        Local file path or HuggingFace dataset ID (e.g. tatsu-lab/alpaca).
      </p>

      <div className="field">
        <label>Dataset source</label>
        <input
          type="text"
          placeholder="/Users/you/data/dataset.jsonl  or  owner/dataset-name"
          value={state.datasetPath}
          onChange={(e) => {
            onChange({ datasetPath: e.target.value });
            setError("");
          }}
        />
      </div>

      {showHfToken && (
        <div className="field">
          <label>HuggingFace token (required for HF datasets)</label>
          <input
            type="password"
            placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
            value={state.hfToken}
            onChange={(e) => onChange({ hfToken: e.target.value })}
          />
        </div>
      )}

      <div className="field">
        <label>File format</label>
        <div className="radio-group">
          {(["jsonl", "json", "csv", "text"] as const).map((fmt) => (
            <label key={fmt} className="radio-option">
              <input
                type="radio"
                name="format"
                checked={state.datasetFormat === fmt}
                onChange={() => onChange({ datasetFormat: fmt })}
              />
              {fmt.toUpperCase()}
              {fmt === "jsonl" && " — recommended"}
            </label>
          ))}
        </div>
      </div>

      <details className="collapsible">
        <summary>Advanced options</summary>
        <div className="content">
          <div className="grid-2">
            <div className="field">
              <label>Text column name</label>
              <input
                type="text"
                value={state.textColumn}
                onChange={(e) => onChange({ textColumn: e.target.value })}
              />
            </div>
            <div className="field">
              <label>Max sequence length</label>
              <input
                type="number"
                value={state.maxSeqLength}
                onChange={(e) => onChange({ maxSeqLength: parseInt(e.target.value) || 512 })}
              />
            </div>
          </div>
        </div>
      </details>

      <div className="bottom-nav">
        <button className="btn" onClick={onBack}>← Back</button>
        <span className="status-msg">{error}</span>
        <button className="btn btn-primary" onClick={handleNext}>Next →</button>
      </div>
    </div>
  );
}
