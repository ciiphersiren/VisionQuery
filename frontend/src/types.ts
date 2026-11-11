export type Mood = "confident" | "hedged" | "stumped" | "curious";
export type Pipeline = "fi1280" | "fi1280+fi1920" | "fi1024" | "sliced";

export interface ModelInfo {
  id: string;
  name: string;
  available: boolean;
  missing: string[];
  classes: string[];
  pipelines: Pipeline[];
  default_pipeline: Pipeline;
  trained_on: string;
  notes: string;
}

export interface Evidence {
  box: [number, number, number, number];
  label: string;
  conf: number;
  target: string;
  strength: "counted" | "weak" | "open";
}

export interface GroupResult {
  label: string;
  classes: string[];
  count: number;
  count_range: [number, number];
  exists: boolean;
  per_class: Record<string, number>;
}

export interface TraceRow {
  class: string;
  count_threshold: number;
  exist_threshold: number;
  detected: number;
  in_region: number;
  counted: number;
  weak: number;
  below_threshold: number;
  open_vocab?: boolean;
}

export interface Answer {
  text: string;
  intent: string;
  value: unknown;
  region: string;
  region_box: [number, number, number, number];
  groups: GroupResult[];
  evidence: Evidence[];
  caveats: string[];
  oov: { term: string; nearest: string[] }[];
  trace: TraceRow[];
  threshold_mode: "calibrated" | "override";
  open_vocab: OpenVocabHit[];
  mood: Mood;
}

export interface OpenVocabHit {
  term: string;
  mode: "novel" | "subtype";
  count: number;
  threshold: number;
  model?: string;
  prompts?: string[];
  parent?: string[];
  parent_count?: number;
}

export interface Plan {
  intent: string;
  targets: { label: string; classes: string[] }[];
  region: string;
  oov: { term: string; nearest: string[] }[];
  source: "rules" | "llm" | "none";
  query: string;
  notes: string[];
}

export interface AskResponse {
  image_id: string;
  plan: Plan;
  answer: Answer;
  detector: { pipeline: Pipeline; latency_ms: number; model?: string };
  latency_ms: number;
}

export interface Meta {
  classes: { name: string; emoji: string }[];
  pipelines: Pipeline[];
  default_pipeline: Pipeline;
  models?: ModelInfo[];
  default_model?: string;
  planner_policy: string;
  llm_available: boolean;
  examples: string[];
  calibration: Record<string, { count_thresholds: Record<string, number>; exist_thresholds: Record<string, number>; exist_recall: Record<string, number> }>;
}

export interface ImageInfo {
  image_id: string;
  width: number;
  height: number;
  src: string;
  name: string;
}

export interface Turn {
  id: number;
  query: string;
  model: string;
  pipeline: Pipeline;
  threshold: number | null;
  status: "thinking" | "done" | "error";
  response?: AskResponse;
  error?: string;
}
