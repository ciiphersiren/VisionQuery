import { useEffect, useRef, useState } from "react";
import { ALL_SCENES, ArtSvg } from "../math/art";

/* Every drawing in the UI, with the equations behind it, ready for Desmos. */
export function Equations({ onClose }: { onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    const d = ref.current;
    if (d && !d.open) d.showModal();
    const close = () => onClose();
    d?.addEventListener("close", close);
    return () => d?.removeEventListener("close", close);
  }, [onClose]);

  const copy = async (name: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(name);
      setTimeout(() => setCopied(null), 1800);
    } catch {
      setCopied("blocked");
    }
  };

  return (
    <dialog ref={ref} className="eq" onClick={(e) => e.target === ref.current && ref.current?.close()}>
      <header className="eq-head">
        <h2>Drawn from equations</h2>
        <button className="link" onClick={() => ref.current?.close()}>
          Close
        </button>
      </header>
      <p className="eq-lede">
        Every flower, leaf and the drone in VisionQuery is a parametric or implicit curve. Copy a drawing and paste its lines into{" "}
        <a href="https://www.desmos.com/calculator" target="_blank" rel="noreferrer">
          Desmos
        </a>
        , one expression per line. Desmos uses y up, so the drawings appear the right way round.
      </p>
      {copied === "blocked" && <p className="eq-lede">The browser blocked clipboard access. Select the expressions below and copy them by hand.</p>}
      {ALL_SCENES.map((s) => (
        <section key={s.name} className="eq-scene">
          <ArtSvg scene={s} className="eq-preview" interactive />
          <div className="eq-body">
            <div className="eq-row">
              <h3>{s.name}</h3>
              <button className="link" onClick={() => copy(s.name, s.parts.map((p) => p.c.desmos).join("\n"))}>
                {copied === s.name ? "Copied" : `Copy ${s.parts.length} expressions for Desmos`}
              </button>
            </div>
            <ul>
              {s.parts.map((p, i) => (
                <li key={i}>
                  <span className="eq-label">{p.c.label}</span> <span className="eq-plain">{p.c.plain}</span>
                </li>
              ))}
            </ul>
          </div>
        </section>
      ))}
    </dialog>
  );
}
