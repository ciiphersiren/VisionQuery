import { useState } from "react";
import { useDictation } from "../hooks/useSpeech";

interface Props {
  disabled: boolean;
  onAsk: (q: string) => void;
}

export function Composer({ disabled, onAsk }: Props) {
  const [q, setQ] = useState("");
  const dict = useDictation((text) => {
    setQ("");
    onAsk(text);
  });

  const submit = (e?: React.FormEvent) => {
    e?.preventDefault();
    const t = q.trim();
    if (!t || disabled) return;
    onAsk(t);
    setQ("");
  };

  return (
    <form className="composer" onSubmit={submit}>
      <label htmlFor="q" className="sr-only">
        Ask about this frame
      </label>
      <input
        id="q"
        value={dict.listening ? dict.interim || q : q}
        onChange={(e) => setQ(e.target.value)}
        placeholder={disabled ? "Choose a photo first" : dict.listening ? "Listening…" : "Ask about this frame"}
        disabled={disabled}
        autoComplete="off"
      />
      {dict.supported && (
        <button
          type="button"
          className={`mic${dict.listening ? " is-on" : ""}`}
          onClick={() => (dict.listening ? dict.stop() : dict.start())}
          disabled={disabled}
          aria-pressed={dict.listening}
          title={dict.listening ? "Stop listening" : "Ask by voice"}
        >
          <svg viewBox="0 0 24 24" aria-hidden width="18" height="18">
            <rect x="9" y="3" width="6" height="11" rx="3" />
            <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7" />
          </svg>
          <span className="sr-only">{dict.listening ? "Stop listening" : "Ask by voice"}</span>
        </button>
      )}
      <button type="submit" className="ask" disabled={disabled || !q.trim()}>
        Ask
      </button>
      {dict.error && <p className="composer-error">{dict.error}</p>}
    </form>
  );
}
