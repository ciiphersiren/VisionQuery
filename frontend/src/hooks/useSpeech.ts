import { useCallback, useEffect, useRef, useState } from "react";

/* Browser speech: dictation via SpeechRecognition (Chrome, Edge, Safari) and
   spoken replies via speechSynthesis (all modern browsers). Nothing leaves the
   browser except what the browser's own recogniser sends. */

type Rec = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((e: any) => void) | null;
  onend: (() => void) | null;
  onerror: ((e: any) => void) | null;
};

const RecCtor: (new () => Rec) | undefined =
  typeof window !== "undefined" ? (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition : undefined;

export function useDictation(onFinal: (text: string) => void) {
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);
  const rec = useRef<Rec | null>(null);
  const finalCb = useRef(onFinal);
  finalCb.current = onFinal;

  const stop = useCallback(() => rec.current?.stop(), []);

  const start = useCallback(() => {
    if (!RecCtor) return;
    setError(null);
    const r = new RecCtor();
    r.lang = navigator.language || "en-US";
    r.interimResults = true;
    r.continuous = false;
    r.onresult = (e: any) => {
      let fin = "";
      let mid = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) fin += t;
        else mid += t;
      }
      setInterim(mid);
      if (fin.trim()) finalCb.current(fin.trim());
    };
    r.onerror = (e: any) => {
      setError(e?.error === "not-allowed" ? "Microphone access is blocked. Allow it in the browser's site settings." : e?.error === "no-speech" ? "I didn't hear anything. Try again a little closer to the mic." : "Dictation stopped unexpectedly. Type your question instead.");
    };
    r.onend = () => {
      setListening(false);
      setInterim("");
    };
    rec.current = r;
    r.start();
    setListening(true);
  }, []);

  useEffect(() => () => rec.current?.abort(), []);

  return { supported: !!RecCtor, listening, interim, error, start, stop };
}

export function speakable(text: string): string {
  return text
    .replace(/\*\*?([^*]+)\*\*?/g, "$1")
    .replace(/\p{Extended_Pictographic}|️|‍/gu, "")
    .replace(/(\d+)\s*–\s*(\d+)/g, "$1 to $2")
    .replace(/\s+/g, " ")
    .trim();
}

export function useVoiceReply() {
  const supported = typeof window !== "undefined" && "speechSynthesis" in window;
  const [speaking, setSpeaking] = useState(false);

  const speak = useCallback(
    (text: string) => {
      if (!supported) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(speakable(text));
      const voices = window.speechSynthesis.getVoices();
      const lang = navigator.language?.slice(0, 2) || "en";
      const v = voices.find((x) => x.lang.startsWith(lang) && /natural|neural|samantha|google/i.test(x.name)) || voices.find((x) => x.lang.startsWith(lang));
      if (v) u.voice = v;
      u.rate = 0.98;
      u.pitch = 1.0;
      u.onstart = () => setSpeaking(true);
      u.onend = () => setSpeaking(false);
      u.onerror = () => setSpeaking(false);
      window.speechSynthesis.speak(u);
    },
    [supported],
  );

  const cancel = useCallback(() => {
    if (supported) window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  return { supported, speaking, speak, cancel };
}
