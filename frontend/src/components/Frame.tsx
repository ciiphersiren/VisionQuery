import { useRef, useState } from "react";
import { FRIENDLY, FRIENDLY_PLURAL, SLOTS, slotFor } from "../palette";
import type { Answer, ImageInfo } from "../types";
import { Bloom } from "./Bloom";
import { ArtSvg, DRONE, SPRIG } from "../math/art";
import { Equations } from "./Equations";

interface Props {
  image: ImageInfo | null;
  answer: Answer | null;
  answerKey: number;
  detecting: boolean;
  onFile: (f: File) => void;
}

/* The pinned drone frame: hand-ruled frame lines that run past the corners,
   the region of the question washed in, and the counted boxes inked on. */
export function Frame({ image, answer, answerKey, detecting, onFile }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const [showEq, setShowEq] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const drop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  };

  const ev = answer?.evidence ?? [];
  const showLabels = ev.length > 0 && ev.length <= 14;
  const W = image?.width ?? 16;
  const H = image?.height ?? 9;
  const rb = answer?.region_box;
  const whole = !answer || answer.region === "whole";
  // summary answers: one colour for everything (class shown in labels); otherwise one colour per target
  const summary = answer?.intent === "summary";
  const openHits = answer?.open_vocab ?? [];
  const targets = summary ? [] : [...(answer?.groups ?? []).map((g) => g.label), ...openHits.map((o) => o.term)];
  const colour = (target: string) => (summary ? SLOTS[0] : slotFor(Math.max(0, targets.indexOf(target))));

  return (
    <figure
      className={`frame${dragging ? " is-dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={drop}
    >
      <div className="frame-mat">
      <span className="frame-rule frame-rule-top" aria-hidden />
      <span className="frame-rule frame-rule-bottom" aria-hidden />
      <span className="frame-rule frame-rule-left" aria-hidden />
      <span className="frame-rule frame-rule-right" aria-hidden />
      <ArtSvg scene={SPRIG} className="sprig" interactive />
      <ArtSvg scene={DRONE} className={`drone${detecting ? " is-busy" : ""}`} interactive />

      {!image ? (
        <button className="frame-empty" onClick={() => input.current?.click()} style={{ aspectRatio: "16 / 10" }}>
          <span className="frame-empty-title">Drop a drone photo here</span>
          <span className="frame-empty-sub">or click to choose one, or pick a sample frame below</span>
        </button>
      ) : (
        <div className="frame-photo" style={{ aspectRatio: `${W} / ${H}` }}>
          <img src={image.src} alt={`Drone frame ${image.name}`} draggable={false} />
          <svg className="overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" key={answerKey}>
            {!whole && rb && (
              <>
                <mask id="region-hole">
                  <rect x="0" y="0" width={W} height={H} fill="white" />
                  <rect x={rb[0]} y={rb[1]} width={rb[2] - rb[0]} height={rb[3] - rb[1]} fill="black" />
                </mask>
                <rect className="region-veil" x="0" y="0" width={W} height={H} mask="url(#region-hole)" />
                <rect className="region-edge" x={rb[0]} y={rb[1]} width={rb[2] - rb[0]} height={rb[3] - rb[1]} vectorEffect="non-scaling-stroke" pathLength={1} />
              </>
            )}
            {ev.map((e, i) => {
              const [x1, y1, x2, y2] = e.box;
              const c = colour(e.target);
              const delay = Math.min(i * 14, 700);
              return (
                <g
                  key={i}
                  className={`box box-${e.strength}${hover === i ? " is-hover" : ""}`}
                  style={{ ["--c" as string]: c, ["--d" as string]: `${delay}ms` }}
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(null)}
                >
                  <rect className="box-halo" x={x1} y={y1} width={x2 - x1} height={y2 - y1} vectorEffect="non-scaling-stroke" />
                  <rect className="box-ink" x={x1} y={y1} width={x2 - x1} height={y2 - y1} vectorEffect="non-scaling-stroke" pathLength={1} />
                  <title>{`${FRIENDLY[e.label] ?? e.label}, ${Math.round(e.conf * 100)}% sure${e.strength === "weak" ? ", below the counting threshold" : ""}`}</title>
                </g>
              );
            })}
          </svg>
          {ev.map((e, i) =>
            showLabels || hover === i ? (
              <span
                key={`l${i}`}
                className={`box-label${hover === i ? " is-hover" : ""}${e.box[0] / W > 0.8 ? " is-flipped" : ""}`}
                style={{
                  left: `${((e.box[0] / W > 0.8 ? e.box[2] : e.box[0]) / W) * 100}%`,
                  top: `${(e.box[1] / H) * 100}%`,
                  ["--c" as string]: colour(e.target),
                }}
              >
                {FRIENDLY[e.label] ?? e.label} {Math.round(e.conf * 100)}%
              </span>
            ) : null,
          )}
          {detecting && (
            <div className="frame-looking" role="status">
              <Bloom mood="thinking" size={30} />
              <span>Looking closely…</span>
            </div>
          )}
        </div>
      )}
      </div>
      <input
        ref={input}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
      {answer && (answer.groups.some((g) => g.count > 0) || openHits.some((o) => o.count > 0)) && (
        <ul className="legend" aria-label="What the boxes show">
          {(summary ? answer.groups.slice(0, 10) : answer.groups).map((g, i) => g.count === 0 ? null : (
            <li key={g.label}>
              <i className="swatch" style={{ background: summary ? SLOTS[0] : slotFor(i) }} />
              {g.count} {summary ? (g.count === 1 ? FRIENDLY[g.label] : FRIENDLY_PLURAL[g.label]) : g.label}
            </li>
          ))}
          {openHits.map((o) =>
            o.count === 0 ? null : (
              <li key={`open-${o.term}`} className="legend-open">
                <i className="swatch swatch-open" style={{ color: slotFor(targets.indexOf(o.term)) }} />
                {o.count} {o.term}, open-vocabulary guess
              </li>
            ),
          )}
        </ul>
      )}
      {image && (
        <figcaption className="frame-caption">
          <span>{image.name}</span>
          <span>
            {image.width} × {image.height}
          </span>
          <button className="link" onClick={() => setShowEq(true)}>
            Drawn from equations
          </button>
          <button className="link" onClick={() => input.current?.click()}>
            Use another photo
          </button>
        </figcaption>
      )}
      {showEq && <Equations onClose={() => setShowEq(false)} />}
    </figure>
  );
}
