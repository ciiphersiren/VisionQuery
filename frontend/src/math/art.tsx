import type { Mood } from "../types";
import { bloom, bud, circle, droop, droopEnd, ellipse, leaf, parabola, segment, stem, superellipse, type Curve } from "./curves";

/* Scenes built only from the curves in curves.ts. Each part knows its class
   (for ink/wash styling) and its equation (shown on hover, exported to Desmos). */

export type Part = { c: Curve; cls: "ink" | "petal" | "heart" | "leaf" | "bud" | "rotor" | "glass" | "fallen" | "cone" };
export interface Scene {
  name: string;
  viewBox: string;
  parts: Part[];
}

const P = (c: Curve, cls: Part["cls"]): Part => ({ c, cls });

function flowerHead(cx: number, cy: number, a: number, opts: Parameters<typeof bloom>[3] = {}): Part[] {
  return [P(bloom(cx, cy, a, opts), "petal"), P(circle(cx, cy, a * 0.3), "heart")];
}

/* Wildflowers growing over the frame's lower-left corner. */
export const SPRIG: Scene = (() => {
  const parts: Part[] = [
    P(stem(40, 230, 162, -7, -2), "ink"),
    P(stem(58, 230, 92, 9, 24), "ink"),
    P(stem(24, 230, 64, -6, -14), "ink"),
    P(stem(40, 230, 128, -7, -10), "ink"),
    P(leaf(37, 150, 30, 7, 200), "leaf"),
    P(leaf(40, 190, 28, 6, -32), "leaf"),
    P(leaf(66, 196, 30, 6, -40), "leaf"),
    P(leaf(18, 200, 24, 5, 230), "leaf"),
    ...flowerHead(38, 62, 19, { p: 0.55, spin: 8 }),
    ...flowerHead(82, 134, 14, { p: 0.5, tilt: 0.82, spin: -14 }),
    ...flowerHead(10, 162, 11.5, { p: 0.6, spin: 20 }),
    P(bud(30, 100, 4.2, 8.5, 18), "bud"),
  ];
  return { name: "Wildflowers", viewBox: "0 0 140 230", parts };
})();

/* An original quadcopter, side view: superellipse body, elliptical rotor discs,
   parabolic landing legs, a circular camera and a faint view cone. */
export const DRONE: Scene = (() => {
  const parts: Part[] = [
    P(segment(80, 64, 54, 112, "view cone, left edge"), "cone"),
    P(segment(80, 64, 112, 112, "view cone, right edge"), "cone"),
    P(segment(60, 36, 30, 27, "arm"), "ink"),
    P(segment(100, 36, 130, 27, "arm"), "ink"),
    P(segment(30, 27, 30, 23, "rotor mast"), "ink"),
    P(segment(130, 27, 130, 23, "rotor mast"), "ink"),
    P(ellipse(30, 22, 25, 3.2, "rotor disc"), "rotor"),
    P(ellipse(130, 22, 25, 3.2, "rotor disc"), "rotor"),
    P(superellipse(80, 40, 23, 9.5, 4, "body"), "glass"),
    P(parabola(66, 48, 0.16, 56, "landing leg"), "ink"),
    P(parabola(94, 48, 0.16, 104, "landing leg"), "ink"),
    P(segment(48, 64, 64, 64, "skid"), "ink"),
    P(segment(96, 64, 112, 64, "skid"), "ink"),
    P(segment(80, 49.5, 80, 52, "gimbal"), "ink"),
    P(circle(80, 56, 4.4), "glass"),
    P(circle(80, 56, 1.8), "ink"),
  ];
  return { name: "Drone", viewBox: "0 0 160 116", parts };
})();

/* Mood flowers, 60 × 80. */
export function moodScene(mood: Mood | "thinking"): Scene {
  let parts: Part[];
  if (mood === "confident") {
    parts = [
      P(stem(30, 80, 50, -1.5), "ink"),
      P(leaf(30, 60, 17, 5.5, 205), "leaf"),
      P(leaf(30, 52, 15, 5, -28), "leaf"),
      ...flowerHead(30, 22, 14, { p: 0.55 }),
    ];
  } else if (mood === "hedged") {
    parts = [
      P(stem(30, 80, 50, 1.5, 7), "ink"),
      P(leaf(31, 58, 16, 5, 205), "leaf"),
      ...flowerHead(37, 24, 12.5, { p: 0.6, tilt: 0.55, spin: 24 }),
    ];
  } else if (mood === "stumped") {
    const [hx, hy] = droopEnd(32, 80, 52, -17);
    parts = [
      P(droop(32, 80, 52, -17), "ink"),
      P(leaf(32, 62, 16, 5, -30), "leaf"),
      P(bloom(hx, hy + 5, 10.5, { p: 0.6, tilt: 0.5, spin: 168 }), "fallen"),
      P(circle(hx, hy + 5, 3), "heart"),
      P(bud(46, 77, 4.2, 1.8, -18), "fallen"),
    ];
  } else {
    parts = [
      P(stem(30, 80, 44, 1.2), "ink"),
      P(leaf(30, 60, 15, 5, 205), "leaf"),
      P(bud(30, 26, 6.2, 11, 0), "bud"),
      P(leaf(30, 37, 7, 2.4, -118), "leaf"),
      P(leaf(30, 37, 7, 2.4, -62), "leaf"),
    ];
  }
  return { name: `Mood flower (${mood})`, viewBox: "0 0 60 80", parts };
}

export function ArtSvg({ scene, className, title, interactive = false, width, height }: { scene: Scene; className?: string; title?: string; interactive?: boolean; width?: number; height?: number }) {
  return (
    <svg className={className} viewBox={scene.viewBox} width={width} height={height} role={title ? "img" : undefined} aria-hidden={title ? undefined : true} aria-label={title}>
      {title && <title>{title}</title>}
      <g filter="url(#ink)">
        {scene.parts.map((p, i) => (
          <path key={i} d={p.c.d} className={`art art-${p.cls}${interactive ? " art-hover" : ""}`}>
            {interactive && <title>{`${p.c.label}: ${p.c.plain}`}</title>}
          </path>
        ))}
      </g>
    </svg>
  );
}

export const ALL_SCENES: Scene[] = [SPRIG, DRONE, moodScene("confident"), moodScene("hedged"), moodScene("stumped"), moodScene("curious")];
