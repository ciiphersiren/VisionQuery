import { Fragment, useState } from "react";
import { FRIENDLY } from "../palette";
import type { Turn } from "../types";
import { Bloom } from "./Bloom";

/* The API decorates answers with emoji for the CLI; the drawn bloom carries mood here. */
const stripEmoji = (t: string) => t.replace(/[\p{Extended_Pictographic}\uFE0F\u200D]+\s*/gu, "").trim();

/* Minimal inline markdown: **bold** and *italic* only. */
function Rich({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("**") ? <strong key={i}>{p.slice(2, -2)}</strong> : p.startsWith("*") && p.length > 2 ? <em key={i}>{p.slice(1, -1)}</em> : <Fragment key={i}>{p}</Fragment>,
      )}
    </>
  );
}

const INTENT: Record<string, string> = {
  count: "count",
  exists: "check if present",
  locate: "find and highlight",
  compare: "compare",
  summary: "list everything",
};

const PIPE: Record<string, string> = { fi1280: "quick look", "fi1280+fi1920": "closer look", fi1024: "whole image", sliced: "sliced" };

interface Props {
  turn: Turn;
  active: boolean;
  onSelect: () => void;
  onSpeak: (t: string) => void;
  speechSupported: boolean;
  modelName?: string;
}

export function Note({ turn, active, onSelect, onSpeak, speechSupported, modelName }: Props) {
  const [open, setOpen] = useState(false);
  const r = turn.response;
  const a = r?.answer;

  return (
    <article className={`note${active ? " is-active" : ""}${turn.status === "thinking" && r ? " is-refreshing" : ""}`} onClick={onSelect}>
      <p className="note-q">{turn.query}</p>

      {turn.status === "thinking" && !r && (
        <div className="note-a">
          <Bloom mood="thinking" size={28} />
          <p className="note-pending">Reading the frame…</p>
        </div>
      )}

      {turn.status === "error" && (
        <div className="note-a">
          <Bloom mood="curious" size={28} />
          <p className="note-error">{turn.error}</p>
        </div>
      )}

      {a && r && (
        <>
          <div className="note-a">
            <Bloom mood={a.mood} size={36} />
            <div className="note-body">
              <p className="note-text">
                <Rich text={stripEmoji(a.text)} />
              </p>
              {a.caveats.map((c, i) => (
                <p key={i} className="note-caveat">
                  {c}
                </p>
              ))}
              <div className="note-meta">
                {modelName && <span>{modelName}</span>}
                <span>{PIPE[r.detector.pipeline] ?? r.detector.pipeline}</span>
                {a.threshold_mode === "override" && turn.threshold != null && <span>threshold {turn.threshold.toFixed(2)}</span>}
                <span>{r.latency_ms < 50 ? "instant (cached)" : `${(r.latency_ms / 1000).toFixed(1)} s`}</span>
                <span className="note-actions">
                  {speechSupported && (
                    <button
                      className="link"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSpeak(a.text);
                      }}
                    >
                      Read aloud
                    </button>
                  )}
                  <button
                    className="link"
                    aria-expanded={open}
                    onClick={(e) => {
                      e.stopPropagation();
                      setOpen((o) => !o);
                    }}
                  >
                    {open ? "Hide how I answered" : "How I answered"}
                  </button>
                </span>
              </div>
            </div>
          </div>
          {open && <Trace turn={turn} />}
        </>
      )}
    </article>
  );
}

function Trace({ turn }: { turn: Turn }) {
  const r = turn.response!;
  const { plan, answer } = r;
  const rows = answer.trace.filter((t) => t.detected > 0 || plan.targets.some((g) => g.classes.includes(t.class)));
  return (
    <div className="trace" onClick={(e) => e.stopPropagation()}>
      <ol className="trace-steps">
        <li>
          <span className="trace-k">Understood as</span>
          <span>
            {INTENT[plan.intent] ?? plan.intent}
            {plan.targets.length > 0 ? <> {plan.targets.map((t) => t.label).join(" and ")}</> : plan.oov.length > 0 && <> {plan.oov.map((o) => o.term).join(" and ")}</>}
            {plan.region !== "whole" && <> in the {plan.region}</>}
            <span className="trace-src"> ({plan.source === "llm" ? "language model" : plan.source === "rules" ? "rules" : "not understood"})</span>
          </span>
        </li>
        {plan.oov.length > 0 && (
          <li>
            <span className="trace-k">Can't see</span>
            <span>
              {plan.oov.map((o) => o.term).join(", ")}
              {answer.open_vocab?.length ? <> — not among my 10 trained classes, so I searched for it by name</> : plan.oov.some((o) => o.nearest.length) && <> — answered with the nearest classes I know</>}
            </span>
          </li>
        )}
        {answer.open_vocab?.map((o) => (
          <li key={o.term}>
            <span className="trace-k">Open vocabulary</span>
            <span>
              {o.model ?? "YOLOE"} prompted with {(o.prompts ?? [o.term]).map((p) => `“${p}”`).join(", ")}; boxes above {o.threshold.toFixed(2)} count.
              {o.mode === "subtype" ? ` Only ${o.parent?.join("/")} boxes from the trained detector that it labels “${o.term}” are counted.` : ""} Zero-shot, not calibrated for drone images.
            </span>
          </li>
        ))}
        <li>
          <span className="trace-k">Detector</span>
          <span>
            {PIPE[r.detector.pipeline]}, boxes {answer.threshold_mode === "override" ? "above your slider value" : "above each class's calibrated threshold"}
          </span>
        </li>
      </ol>
      {rows.length > 0 && (
        <table className="trace-table">
          <caption className="sr-only">Boxes kept and dropped per class</caption>
          <thead>
            <tr>
              <th scope="col">Class</th>
              <th scope="col" title="Count threshold">Counts at</th>
              <th scope="col">Seen</th>
              <th scope="col">In region</th>
              <th scope="col">Counted</th>
              <th scope="col" title="Above the existence threshold but below the counting one">Faint</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((t) => (
              <tr key={t.class}>
                <th scope="row">
                  {FRIENDLY[t.class] ?? `${t.class} (open vocabulary)`}
                </th>
                <td>{t.count_threshold.toFixed(2)}</td>
                <td>{t.detected}</td>
                <td>{t.in_region}</td>
                <td className="strong">{t.counted}</td>
                <td>{t.weak}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
