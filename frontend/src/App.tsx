import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { Composer } from "./components/Composer";
import { Controls } from "./components/Controls";
import { Frame } from "./components/Frame";
import { Metrics } from "./components/Metrics";
import { Note } from "./components/Note";
import { useVoiceReply } from "./hooks/useSpeech";
import type { ImageInfo, Meta, ModelInfo, Pipeline, Turn } from "./types";

type View = "ask" | "record";
type Theme = "auto" | "light" | "dark";

function readPref<T>(k: string, d: T): T {
  try {
    const v = localStorage.getItem(k);
    return v == null ? d : (JSON.parse(v) as T);
  } catch {
    return d;
  }
}
function writePref(k: string, v: unknown) {
  try {
    localStorage.setItem(k, JSON.stringify(v));
  } catch {
    /* private mode */
  }
}

export default function App() {
  const [view, setView] = useState<View>(() => (location.hash === "#record" ? "record" : "ask"));
  const [meta, setMeta] = useState<Meta | null>(null);
  const [samples, setSamples] = useState<{ name: string; url: string }[]>([]);
  const [image, setImage] = useState<ImageInfo | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [pipeline, setPipeline] = useState<Pipeline>(() => readPref("sq.pipeline", "fi1280"));
  const [model, setModel] = useState<string>(() => readPref("sq.model", ""));
  const [threshold, setThreshold] = useState<number | null>(null);
  const [voice, setVoice] = useState<boolean>(() => readPref("sq.voice", false));
  const [theme, setTheme] = useState<Theme>(() => readPref("sq.theme", "auto"));
  const [connected, setConnected] = useState<boolean | null>(null);
  const [seen, setSeen] = useState<Set<string>>(new Set());
  const [loadError, setLoadError] = useState<string | null>(null);
  const nextId = useRef(1);
  const notesEnd = useRef<HTMLDivElement>(null);
  const tts = useVoiceReply();

  // Keep trying until the backend answers, so starting the UI before `skyquery serve` just works.
  useEffect(() => {
    let stop = false;
    let timer: number | undefined;
    const tryConnect = async () => {
      try {
        const m = await api.meta();
        if (stop) return;
        setMeta(m);
        setConnected(true);
        api.samples().then((s) => !stop && setSamples(s)).catch(() => {});
      } catch {
        if (stop) return;
        setConnected(false);
        timer = window.setTimeout(tryConnect, 2500);
      }
    };
    tryConnect();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, []);
  useEffect(() => {
    writePref("sq.theme", theme);
    const root = document.documentElement;
    if (theme === "auto") delete root.dataset.theme;
    else root.dataset.theme = theme;
  }, [theme]);
  useEffect(() => writePref("sq.pipeline", pipeline), [pipeline]);
  useEffect(() => writePref("sq.model", model), [model]);
  const models: ModelInfo[] = meta?.models ?? [];
  // A remembered model that is gone or not installed falls back to the server's default; the pipeline follows the model.
  useEffect(() => {
    if (!models.length) return;
    const m = models.find((x) => x.id === model && x.available) ?? models.find((x) => x.id === meta?.default_model) ?? models[0];
    if (m.id !== model) setModel(m.id);
    if (!m.pipelines.includes(pipeline)) setPipeline(m.default_pipeline);
  }, [models, model, pipeline, meta?.default_model]);
  useEffect(() => writePref("sq.voice", voice), [voice]);
  useEffect(() => {
    const h = () => setView(location.hash === "#record" ? "record" : "ask");
    addEventListener("hashchange", h);
    return () => removeEventListener("hashchange", h);
  }, []);

  const active = turns.find((t) => t.id === activeId) ?? null;
  const detecting = turns.some((t) => t.status === "thinking") && image != null && !seen.has(`${image.image_id}|${model}|${pipeline}`);

  const run = useCallback(
    async (turn: Turn, img: ImageInfo, replace: boolean) => {
      setTurns((ts) => (replace ? ts.map((t) => (t.id === turn.id ? turn : t)) : [...ts, turn]));
      setActiveId(turn.id);
      try {
        const res = await api.ask(img.image_id, turn.query, turn.pipeline, turn.threshold, turn.model);
        setSeen((s) => new Set(s).add(`${img.image_id}|${turn.model}|${turn.pipeline}`));
        setTurns((ts) => ts.map((t) => (t.id === turn.id ? { ...turn, status: "done", response: res } : t)));
        return res;
      } catch (e) {
        const msg = (e as Error).message.includes("unknown image_id")
          ? "The server restarted and forgot this photo. Choose it again to keep asking."
          : `That question failed: ${(e as Error).message}`;
        setTurns((ts) => ts.map((t) => (t.id === turn.id ? { ...turn, status: "error", error: msg } : t)));
        return null;
      }
    },
    [],
  );

  const ask = useCallback(
    async (query: string) => {
      if (!image) return;
      const turn: Turn = { id: nextId.current++, query, model, pipeline, threshold, status: "thinking" };
      const res = await run(turn, image, false);
      if (res && voice) tts.speak(res.answer.text);
    },
    [image, model, pipeline, threshold, run, voice, tts],
  );

  // Slider and detail toggle re-answer the active note in place (detections are cached, so this is quick).
  const rerunTimer = useRef<number | undefined>(undefined);
  const rerunActive = useCallback(
    (patch: Partial<Pick<Turn, "model" | "pipeline" | "threshold">>, delay = 0) => {
      if (!active || !image || active.status === "thinking") return;
      window.clearTimeout(rerunTimer.current);
      rerunTimer.current = window.setTimeout(() => {
        run({ ...active, ...patch, status: "thinking", response: active.response }, image, true);
      }, delay);
    },
    [active, image, run],
  );

  const choose = async (loader: () => Promise<{ image_id: string; width: number; height: number }>, src: string, name: string) => {
    setLoadError(null);
    try {
      const r = await loader();
      setImage({ ...r, src, name });
      setTurns([]);
      setActiveId(null);
    } catch (e) {
      setLoadError(`Couldn't open that photo: ${(e as Error).message}`);
    }
  };

  const onFile = (f: File) => choose(() => api.upload(f), URL.createObjectURL(f), f.name);

  useEffect(() => {
    const list = notesEnd.current?.parentElement;
    if (list && list.scrollHeight > list.clientHeight) list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }, [turns.length]);

  return (
    <div className="app">
      <InkDefs />
      <header className="masthead">
        <a className="wordmark" href="#" onClick={() => (location.hash = "")}>
          VisionQuery
        </a>
        <p className="tagline">Ask about a drone photo. Every answer is counted from boxes you can see.</p>
        <nav className="views" aria-label="Views">
          <a href="#" aria-current={view === "ask" ? "page" : undefined}>
            Ask
          </a>
          <a href="#record" aria-current={view === "record" ? "page" : undefined}>
            How it works
          </a>
        </nav>
        <fieldset className="seg theme">
          <legend>Theme</legend>
          {(["auto", "light", "dark"] as Theme[]).map((t) => (
            <label key={t} className={theme === t ? "is-on" : ""} title={t === "auto" ? "Follow the system setting" : undefined}>
              <input type="radio" name="theme" value={t} checked={theme === t} onChange={() => setTheme(t)} />
              {t === "auto" ? "Auto" : t === "light" ? "Light" : "Dark"}
            </label>
          ))}
        </fieldset>
      </header>

      {connected === false && (
        <p className="offline" role="status">
          Waiting for the VisionQuery server at 127.0.0.1:8000. Start it with <code>visionquery serve</code> in the project folder; this page
          connects by itself once it's running.
        </p>
      )}
      {view === "record" ? (
        <Metrics />
      ) : (
        <main className="desk">
          <section className="plate" aria-label="Photo">
            <Frame image={image} answer={active?.response?.answer ?? null} answerKey={active?.response ? active.id * 1000 + (active.response.latency_ms | 0) : 0} detecting={detecting} onFile={onFile} />
            {samples.length > 0 && (
              <div className="samples">
                <p className="samples-k">Sample frames from VisDrone</p>
                <ul>
                  {samples.map((s) => (
                    <li key={s.name}>
                      <button
                        className={image?.name === s.name ? "is-on" : ""}
                        onClick={() => choose(() => api.loadSample(`${s.name}.jpg`), s.url, s.name)}
                        aria-label={`Use sample frame ${s.name}`}
                      >
                        <img src={s.url} alt="" loading="lazy" />
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {loadError && <p className="load-error">{loadError}</p>}
          </section>

          <section className="margin" aria-label="Questions and answers">
            <Controls
              models={models}
              model={model}
              onModel={(m) => {
                const spec = models.find((x) => x.id === m);
                const p = spec && !spec.pipelines.includes(pipeline) ? spec.default_pipeline : pipeline;
                setModel(m);
                setPipeline(p);
                rerunActive({ model: m, pipeline: p });
              }}
              pipeline={pipeline}
              onPipeline={(p) => {
                setPipeline(p);
                rerunActive({ pipeline: p });
              }}
              threshold={threshold}
              onThreshold={(t) => {
                setThreshold(t);
                rerunActive({ threshold: t }, 220);
              }}
              voice={voice}
              onVoice={(v) => {
                setVoice(v);
                if (!v) tts.cancel();
              }}
              voiceSupported={tts.supported}
            />

            <div className="notes" aria-live="polite">
              {turns.length === 0 ? (
                <div className="notes-empty">
                  <p>{image ? "Try one of these, or ask your own:" : "Choose a photo, then ask things like:"}</p>
                  <ul>
                    {(meta?.examples ?? []).map((ex) => (
                      <li key={ex}>
                        <button className="example" disabled={!image} onClick={() => ask(ex)}>
                          {ex}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                turns.map((t) => <Note key={t.id} turn={t} active={t.id === activeId} onSelect={() => setActiveId(t.id)} onSpeak={tts.speak} speechSupported={tts.supported} modelName={models.length > 1 ? models.find((m) => m.id === t.model)?.name : undefined} />)
              )}
              <div ref={notesEnd} />
            </div>

            <Composer disabled={!image} onAsk={ask} />
          </section>
        </main>
      )}
    </div>
  );
}

/* Shared SVG filter: a faint wobble that makes drawn lines read as ink. */
function InkDefs() {
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden>
      <filter id="ink" x="-10%" y="-10%" width="120%" height="120%">
        <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="2" seed="7" />
        <feDisplacementMap in="SourceGraphic" scale="2.2" />
      </filter>
    </svg>
  );
}
