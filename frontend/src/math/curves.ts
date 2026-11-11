/* Every drawn shape in the UI comes from an equation.
   Each generator returns an SVG path (screen space, y down) and the same curve
   as a Desmos expression (y up), so the art can be pasted into Desmos as-is. */

export interface Curve {
  d: string;
  desmos: string;
  label: string;
  plain: string;
}

const N = 160;
const f = (v: number) => +v.toFixed(2);
const tex = (v: number) => f(v).toString();
const term = (coef: number, expr: string, digits = 2) => {
  const c = +coef.toFixed(digits);
  if (c === 0) return "";
  return `${c < 0 ? "-" : "+"}${Math.abs(c) === 1 ? "" : Math.abs(c)}\\left(${expr}\\right)`;
};
/* " + 7·sin(πt)" / " − 7·sin(πt)" for readable formulas */
const pm = (v: number, expr = "") => {
  const c = f(v);
  if (c === 0) return "";
  return ` ${c < 0 ? "−" : "+"} ${Math.abs(c)}${expr}`;
};
const T2PI = "\\ \\left\\{0\\le t\\le2\\pi\\right\\}";
/* Desmos (y up) form of a local curve (X(t), Y(t)) placed at screen (x0, y0) and rotated `deg` screen degrees. */
function placed(X: string, Y: string, x0: number, y0: number, deg: number, domain = T2PI): string {
  const a = (-deg * Math.PI) / 180;
  const c = Math.cos(a);
  const sn = Math.sin(a);
  return `\\left(${tex(x0)}${term(c, X, 4)}${term(-sn, Y, 4)},\\ ${tex(-y0)}${term(sn, X, 4)}${term(c, Y, 4)}\\right)${domain}`;
}

function path(fn: (t: number) => [number, number], t0: number, t1: number, closed = false, n = N): string {
  let d = "";
  for (let i = 0; i <= n; i++) {
    const t = t0 + ((t1 - t0) * i) / n;
    const [x, y] = fn(t);
    d += `${i ? "L" : "M"}${f(x)} ${f(y)}`;
  }
  return closed ? d + "Z" : d;
}

/* Rotate (x, y) about (cx, cy) by `deg` degrees (screen space). */
function rot(x: number, y: number, cx: number, cy: number, deg: number): [number, number] {
  const a = (deg * Math.PI) / 180;
  const dx = x - cx;
  const dy = y - cy;
  return [cx + dx * Math.cos(a) - dy * Math.sin(a), cy + dx * Math.sin(a) + dy * Math.cos(a)];
}

/** Five-petal bloom: r(θ) = a·|cos(5θ/2)|^p, squashed by `tilt` for perspective. */
export function bloom(cx: number, cy: number, a: number, opts: { petals?: number; p?: number; tilt?: number; spin?: number } = {}): Curve {
  const k = opts.petals ?? 5;
  const p = opts.p ?? 0.55;
  const tilt = opts.tilt ?? 1;
  const spin = opts.spin ?? 0;
  const r = (t: number) => a * Math.pow(Math.abs(Math.cos((k * t) / 2)), p);
  const d = path(
    (t) => rot(cx + r(t) * Math.cos(t), cy + r(t) * Math.sin(t) * tilt, cx, cy, spin),
    0,
    2 * Math.PI,
    true,
    360,
  );
  const R = `\\left|\\cos\\left(\\frac{${k}t}{2}\\right)\\right|^{${tex(p)}}`;
  return {
    d,
    label: `${k}-petal bloom`,
    plain: `r(θ) = ${tex(a)}·|cos(${k}θ/2)|^${tex(p)}, centred at (${tex(cx)}, ${tex(cy)})${tilt !== 1 ? `, y squashed ×${tex(tilt)}` : ""}${spin ? `, turned ${tex(spin)}°` : ""}`,
    desmos: placed(`${tex(a)}${R}\\cos t`, `${tex(a * tilt)}${R}\\sin t`, cx, cy, spin),
  };
}

/** Flower heart: a circle. */
export function circle(cx: number, cy: number, r: number): Curve {
  return {
    d: path((t) => [cx + r * Math.cos(t), cy + r * Math.sin(t)], 0, 2 * Math.PI, true, 60),
    label: "circle",
    plain: `(x − ${tex(cx)})² + (y − ${tex(cy)})² = ${tex(r)}²`,
    desmos: `\\left(x-${tex(cx)}\\right)^{2}+\\left(y+${tex(cy)}\\right)^{2}=${tex(r * r)}`,
  };
}

/** Stem: x(t) = x0 + A·sin(πt)·w(t), y(t) = y0 − L·t, with a slight lean `b`. */
export function stem(x0: number, y0: number, L: number, A: number, b = 0): Curve {
  return {
    d: path((t) => [x0 + A * Math.sin(Math.PI * t) + b * t, y0 - L * t], 0, 1, false, 60),
    label: "stem",
    plain: `x = ${tex(x0)}${pm(A, "·sin(πt)")}${pm(b, "t")},  y = ${tex(y0)} − ${tex(L)}t,  0 ≤ t ≤ 1`,
    desmos: `\\left(${tex(x0)}${term(A, "\\sin\\left(\\pi t\\right)")}${term(b, "t")},\\ ${tex(-y0)}${term(L, "t")}\\right)\\ \\left\\{0\\le t\\le1\\right\\}`,
  };
}

/** Drooping stem: rises, peaks at t = 2/3, then bows down: x = x0 + b·t², y = y0 − L·sin(3πt/4). */
export function droop(x0: number, y0: number, L: number, bend: number): Curve {
  return {
    d: path((t) => [x0 + bend * t * t, y0 - L * Math.sin((3 * Math.PI * t) / 4)], 0, 1, false, 60),
    label: "drooping stem",
    plain: `x = ${tex(x0)}${pm(bend, "t²")},  y = ${tex(y0)} − ${tex(L)}·sin(3πt/4),  0 ≤ t ≤ 1`,
    desmos: `\\left(${tex(x0)}${term(bend, "t^{2}")},\\ ${tex(-y0)}${term(L, "\\sin\\left(\\frac{3\\pi t}{4}\\right)")}\\right)\\ \\left\\{0\\le t\\le1\\right\\}`,
  };
}

/** End point of a droop() stem, for placing the head. */
export const droopEnd = (x0: number, y0: number, L: number, bend: number): [number, number] => [x0 + bend, y0 - L * Math.sin((3 * Math.PI) / 4)];

/** Leaf: teardrop x = L(1−cos t)/2, y = W·sin t·sin(t/2), rotated by `deg` about its base. */
export function leaf(x0: number, y0: number, L: number, W: number, deg: number): Curve {
  const d = path((t) => rot(x0 + (L * (1 - Math.cos(t))) / 2, y0 + W * Math.sin(t) * Math.sin(t / 2), x0, y0, deg), 0, 2 * Math.PI, true, 120);
  return {
    d,
    label: "leaf",
    plain: `x = ${tex(L)}(1 − cos t)/2,  y = ${tex(W)}·sin t·sin(t/2), from (${tex(x0)}, ${tex(y0)}), turned ${tex(deg)}°`,
    desmos: placed(`\\frac{${tex(L)}\\left(1-\\cos t\\right)}{2}`, `${tex(W)}\\sin t\\sin\\frac{t}{2}`, x0, y0, deg),
  };
}

/** Bud: a narrow ellipse x = a·cos t, y = b·sin t, rotated. */
export function bud(cx: number, cy: number, a: number, b: number, deg = 0): Curve {
  return {
    d: path((t) => rot(cx + a * Math.cos(t), cy + b * Math.sin(t), cx, cy, deg), 0, 2 * Math.PI, true, 80),
    label: "bud",
    plain: `x = ${tex(cx)}${pm(a, "·cos t")},  y = ${tex(cy)}${pm(b, "·sin t")}${deg ? `, turned ${tex(deg)}°` : ""}`,
    desmos: placed(`${tex(a)}\\cos t`, `${tex(b)}\\sin t`, cx, cy, deg),
  };
}

/** Straight segment. */
export function segment(x1: number, y1: number, x2: number, y2: number, label = "segment"): Curve {
  return {
    d: `M${f(x1)} ${f(y1)}L${f(x2)} ${f(y2)}`,
    label,
    plain: `from (${tex(x1)}, ${tex(y1)}) to (${tex(x2)}, ${tex(y2)})`,
    desmos: `\\left(${tex(x1)}${term(x2 - x1, "t")},\\ ${tex(-y1)}${term(-(y2 - y1), "t")}\\right)\\ \\left\\{0\\le t\\le1\\right\\}`,
  };
}

/** Ellipse (rotor discs, drone body). */
export function ellipse(cx: number, cy: number, a: number, b: number, label = "ellipse"): Curve {
  return {
    d: path((t) => [cx + a * Math.cos(t), cy + b * Math.sin(t)], 0, 2 * Math.PI, true, 90),
    label,
    plain: `((x − ${tex(cx)})/${tex(a)})² + ((y − ${tex(cy)})/${tex(b)})² = 1`,
    desmos: `\\frac{\\left(x-${tex(cx)}\\right)^{2}}{${tex(a * a)}}+\\frac{\\left(y+${tex(cy)}\\right)^{2}}{${tex(b * b)}}=1`,
  };
}

/** Superellipse |x/a|^n + |y/b|^n = 1 — the drone's rounded-rectangle body. */
export function superellipse(cx: number, cy: number, a: number, b: number, n: number, label = "body"): Curve {
  const sg = (v: number) => Math.sign(v);
  const e = 2 / n;
  return {
    d: path((t) => [cx + a * sg(Math.cos(t)) * Math.pow(Math.abs(Math.cos(t)), e), cy + b * sg(Math.sin(t)) * Math.pow(Math.abs(Math.sin(t)), e)], 0, 2 * Math.PI, true, 160),
    label,
    plain: `|(x − ${tex(cx)})/${tex(a)}|^${n} + |(y − ${tex(cy)})/${tex(b)}|^${n} = 1`,
    desmos: `\\left|\\frac{x-${tex(cx)}}{${tex(a)}}\\right|^{${n}}+\\left|\\frac{y+${tex(cy)}}{${tex(b)}}\\right|^{${n}}=1`,
  };
}

/** Landing leg: parabola y = y0 + k(x − x0)², for x between x0 and x1. */
export function parabola(x0: number, y0: number, k: number, x1: number, label = "leg"): Curve {
  return {
    d: path((t) => [x0 + (x1 - x0) * t, y0 + k * ((x1 - x0) * t) ** 2], 0, 1, false, 40),
    label,
    plain: `y = ${tex(y0)}${pm(k, `(x − ${tex(x0)})²`)},  x from ${tex(x0)} to ${tex(x1)}`,
    desmos: `y=${tex(-y0)}-${tex(k)}\\left(x-${tex(x0)}\\right)^{2}\\ \\left\\{${tex(Math.min(x0, x1))}\\le x\\le${tex(Math.max(x0, x1))}\\right\\}`,
  };
}
