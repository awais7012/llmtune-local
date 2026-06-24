export type TrainMode = "llm" | "cnn";

export interface ModelInfo {
  id: string;
  label: string;
  size: string;
}

export interface HFModelResult {
  id: string;
  label: string;
  downloads: number;
  size: string | null;
}

export interface WizardState {
  mode: TrainMode;
  modelId: string;
  localPath: string;
  hfToken: string;
  // Dataset
  datasetPath: string;
  datasetFormat: "jsonl" | "json" | "csv" | "text";
  textColumn: string;
  maxSeqLength: number;
  validationSplit: number; // 0 = none, 0.1 = 10%, 0.2 = 20%
  packSequences: boolean;
  shuffleDataset: boolean;
  // Training params
  numEpochs: number;
  batchSize: number;
  gradAccum: number;
  learningRate: number;
  warmupRatio: number;
  warmupSteps: number;
  lrSchedulerType: "cosine" | "linear" | "constant" | "cosine_with_restarts";
  // LoRA
  loraR: number;
  loraAlpha: number;
  loraDropout: number;
  targetModules: string[];
  // Quantization / method
  quantization: "none" | "4bit" | "8bit";
  trainingMode: "feature_extraction" | "full" | "lora";
  // Toggles
  flashAttention: boolean;
  saveCheckpoints: boolean;
  evalOnValidation: boolean;
  resumeFromCheckpoint: boolean;
  pushToHub: boolean;
  hubModelId: string;
  // Output
  outputDir: string;
  // Config management
  configName: string;
}

export const defaultWizardState = (): WizardState => ({
  mode: "llm",
  modelId: "",
  localPath: "",
  hfToken: "",
  datasetPath: "",
  datasetFormat: "jsonl",
  textColumn: "text",
  maxSeqLength: 2048,
  validationSplit: 0,
  packSequences: false,
  shuffleDataset: true,
  numEpochs: 3,
  batchSize: 2,
  gradAccum: 4,
  learningRate: 2e-4,
  warmupRatio: 0.03,
  warmupSteps: 50,
  lrSchedulerType: "cosine",
  loraR: 16,
  loraAlpha: 32,
  loraDropout: 0.05,
  targetModules: ["q_proj", "v_proj"],
  quantization: "none",
  trainingMode: "feature_extraction",
  flashAttention: true,
  saveCheckpoints: true,
  evalOnValidation: true,
  resumeFromCheckpoint: false,
  pushToHub: false,
  hubModelId: "",
  outputDir: "",
  configName: "",
});
