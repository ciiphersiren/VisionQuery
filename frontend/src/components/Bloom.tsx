import { ArtSvg, moodScene } from "../math/art";
import type { Mood } from "../types";

/* A small flower, drawn from equations, that reports how sure an answer is.
   confident = open bloom · hedged = half-open, tilted · stumped = drooping
   curious = closed bud · thinking = bud that sways while the detector runs */

type State = Mood | "thinking";

const LABEL: Record<State, string> = {
  confident: "Sure of this answer",
  hedged: "Fairly sure, with caveats",
  stumped: "Outside what I can see",
  curious: "Needs a clearer question",
  thinking: "Looking…",
};

export function Bloom({ mood, size = 44 }: { mood: State; size?: number }) {
  return (
    <span className={`bloom bloom-${mood}`} style={{ width: size, height: size * 1.333 }}>
      <ArtSvg scene={moodScene(mood === "thinking" ? "curious" : mood)} title={LABEL[mood]} width={size} height={size * 1.333} className={mood === "thinking" ? "bloom-sway" : undefined} />
    </span>
  );
}
