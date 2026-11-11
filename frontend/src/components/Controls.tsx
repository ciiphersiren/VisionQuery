import type { ModelInfo, Pipeline } from "../types";

export const PIPELINE_LABELS: Record<Pipeline, [string, string]> = {
  fi1280: ["Quick look", "One pass at 1280 px"],
  "fi1280+fi1920": ["Closer look", "Two passes, 1280 and 1920 px: finds more small objects, about 3× slower"],
  fi1024: ["Whole image", "One pass at 1024 px, the setup this model was chosen with"],
  sliced: ["Sliced", "512 px tiles enlarged to 1280 plus the whole image at 1536: best for tiny objects, slow on a CPU for a new photo"],
};

interface Props {
  models: ModelInfo[];
  model: string;
  onModel: (m: string) => void;
  pipeline: Pipeline;
  onPipeline: (p: Pipeline) => void;
  threshold: number | null;
  onThreshold: (t: number | null) => void;
  voice: boolean;
  onVoice: (v: boolean) => void;
  voiceSupported: boolean;
}

export function Controls({ models, model, onModel, pipeline, onPipeline, threshold, onThreshold, voice, onVoice, voiceSupported }: Props) {
  const t = threshold ?? 0.3;
  const current = models.find((m) => m.id === model);
  const pipelines = current?.pipelines ?? (["fi1280", "fi1280+fi1920"] as Pipeline[]);
  return (
    <div className="controls">
      {models.length > 1 && (
        <div className="model-pick">
          <label htmlFor="model">Model</label>
          <select id="model" value={model} onChange={(e) => onModel(e.target.value)}>
            {models.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.available}>
                {m.name}
                {m.available ? "" : " (not installed)"}
              </option>
            ))}
          </select>
          {current && (
            <p className="model-note">
              Knows {current.classes.length === 10 ? "all 10 VisDrone classes" : current.classes.join(", ")}. {current.notes}
            </p>
          )}
          {models.some((m) => !m.available) && (
            <details className="model-note">
              <summary>{models.filter((m) => !m.available).length} more need their weights file</summary>
              <ul>
                {models
                  .filter((m) => !m.available)
                  .map((m) => (
                    <li key={m.id}>
                      {m.name}: {m.missing.join(", ")}
                    </li>
                  ))}
              </ul>
              Put the files in skyquery/weights/ and restart the server.
            </details>
          )}
        </div>
      )}

      {pipelines.length > 1 ? (
        <fieldset className="seg">
          <legend>Detail</legend>
          {pipelines.map((v) => (
            <label key={v} title={PIPELINE_LABELS[v][1]} className={pipeline === v ? "is-on" : ""}>
              <input type="radio" name="pipeline" value={v} checked={pipeline === v} onChange={() => onPipeline(v)} />
              {PIPELINE_LABELS[v][0]}
            </label>
          ))}
        </fieldset>
      ) : (
        <p className="model-note">Detail: {PIPELINE_LABELS[pipelines[0]][0].toLowerCase()}. {PIPELINE_LABELS[pipelines[0]][1]}.</p>
      )}

      <div className="slider">
        <label htmlFor="thr">
          Counting threshold
          <span className="slider-val">{threshold == null ? "calibrated per class" : `${Math.round(t * 100)}% for all classes`}</span>
        </label>
        <div className="slider-row">
          <input
            id="thr"
            type="range"
            min={0.05}
            max={0.9}
            step={0.025}
            value={t}
            onChange={(e) => onThreshold(parseFloat(e.target.value))}
            aria-valuetext={threshold == null ? "calibrated" : `${Math.round(t * 100)} percent`}
            className={threshold == null ? "is-calibrated" : ""}
          />
          <button className="link" onClick={() => onThreshold(null)} disabled={threshold == null}>
            Reset
          </button>
        </div>
        <div className="slider-marks" aria-hidden>
          <span>more boxes, more mistakes</span>
          <span>fewer, surer</span>
        </div>
      </div>

      {voiceSupported && (
        <label className="toggle">
          <input type="checkbox" checked={voice} onChange={(e) => onVoice(e.target.checked)} />
          <span className="toggle-track" aria-hidden />
          Read answers aloud
        </label>
      )}
    </div>
  );
}
