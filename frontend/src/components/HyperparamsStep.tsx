import { useState } from "react";
import type { AppConfig } from "../api";
import type { WizardState } from "../types";

interface Props {
  state: WizardState;
  appConfig: AppConfig | null;
  onChange: (patch: Partial<WizardState>) => void;
  onStart: () => void;
  onBack: () => void;
}

export default function HyperparamsStep({ state, appConfig, onChange, onStart, onBack }: Props) {
  const [error, setError] = useState("");

  function handleStart() {
    if (state.numEpochs < 1 || state.batchSize < 1) {
      setError("Epochs and batch size must be at least 1.");
      return;
    }
    setError("");
    onStart();
  }

  const defaultOutput =
    state.mode === "cnn"
      ? appConfig?.default_cnn_output ?? "~/llmtune-cnn-output"
      : appConfig?.default_llm_output ?? "~/llmtune-output";

  return (
    <div>
      <h2 className="screen-title">Training Configuration</h2>
      <p className="screen-subtitle">Adjust hyperparameters or use the defaults.</p>

      <div className="grid-2">
        <div className="field">
          <label>Epochs</label>
          <input
            type="number"
            value={state.numEpochs}
            onChange={(e) => onChange({ numEpochs: parseInt(e.target.value) || 1 })}
          />
        </div>
        <div className="field">
          <label>Batch size</label>
          <input
            type="number"
            value={state.batchSize}
            onChange={(e) => onChange({ batchSize: parseInt(e.target.value) || 1 })}
          />
        </div>
      </div>

      {state.mode === "llm" && (
        <div className="grid-2">
          <div className="field">
            <label>Gradient accumulation</label>
            <input
              type="number"
              value={state.gradAccum}
              onChange={(e) => onChange({ gradAccum: parseInt(e.target.value) || 1 })}
            />
          </div>
          <div className="field">
            <label>Quantization</label>
            <select
              value={state.quantization}
              onChange={(e) =>
                onChange({ quantization: e.target.value as WizardState["quantization"] })
              }
            >
              <option value="none">None — float16 LoRA (Mac, CPU, any GPU)</option>
              <option value="4bit">4-bit QLoRA — NVIDIA CUDA only</option>
              <option value="8bit">8-bit — NVIDIA CUDA only</option>
            </select>
          </div>
        </div>
      )}

      {state.mode === "cnn" && (
        <div className="field">
          <label>Training mode</label>
          <select
            value={state.trainingMode}
            onChange={(e) =>
              onChange({ trainingMode: e.target.value as WizardState["trainingMode"] })
            }
          >
            <option value="feature_extraction">Feature extraction — fastest</option>
            <option value="full">Full fine-tune — most accurate</option>
            <option value="lora">LoRA — good balance (ViT / Swin)</option>
          </select>
        </div>
      )}

      <div className="field">
        <label>Output folder</label>
        <input
          type="text"
          placeholder={defaultOutput}
          value={state.outputDir}
          onChange={(e) => onChange({ outputDir: e.target.value })}
        />
      </div>

      <details className="collapsible">
        <summary>LoRA settings (advanced)</summary>
        <div className="content">
          <div className="grid-2">
            <div className="field">
              <label>Rank (r)</label>
              <input
                type="number"
                value={state.loraR}
                onChange={(e) => onChange({ loraR: parseInt(e.target.value) || 16 })}
              />
            </div>
            <div className="field">
              <label>Alpha</label>
              <input
                type="number"
                value={state.loraAlpha}
                onChange={(e) => onChange({ loraAlpha: parseInt(e.target.value) || 32 })}
              />
            </div>
            <div className="field">
              <label>Dropout</label>
              <input
                type="number"
                step="0.01"
                value={state.loraDropout}
                onChange={(e) => onChange({ loraDropout: parseFloat(e.target.value) || 0.05 })}
              />
            </div>
            <div className="field">
              <label>Learning rate</label>
              <input
                type="text"
                value={state.learningRate}
                onChange={(e) => onChange({ learningRate: parseFloat(e.target.value) || 2e-4 })}
              />
            </div>
          </div>
        </div>
      </details>

      <div className="bottom-nav">
        <button className="btn" onClick={onBack}>← Back</button>
        <span className="status-msg">{error}</span>
        <button className="btn btn-success" onClick={handleStart}>
          Start Fine-tuning →
        </button>
      </div>
    </div>
  );
}
