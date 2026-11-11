import { useEffect, useState } from "react";
import { api } from "../api";
import { FRIENDLY } from "../palette";
import { Guide } from "./Guide";

/* "How well it works": the evaluation reports, in plain language. */

const REPORT_AP: Record<string, Record<string, number>> = {
  "val fi1280": { AP: 34.63, AP50: 55.06, AP75: 35.79, AP_tiny: 15.54, AP_s: 25.63, AP_m: 46.84 },
  "val fi1280+fi1920": { AP: 37.13, AP50: 59.25, AP75: 38.53, AP_tiny: 18.72, AP_s: 28.82, AP_m: 48.36 },
  "testdev fi1280": { AP: 26.83, AP50: 44.71, AP75: 27.25, AP_tiny: 9.89, AP_s: 16.43, AP_m: 38.31 },
  "testdev fi1280+fi1920": { AP: 28.57, AP50: 47.9, AP75: 28.87, AP_tiny: 11.24, AP_s: 18.33, AP_m: 39.58 },
};
const AP_COLS: [string, string][] = [
  ["AP", "AP"],
  ["AP50", "AP at 50% overlap"],
  ["AP75", "AP at 75% overlap"],
  ["AP_tiny", "Tiny (< 16 px)"],
  ["AP_s", "Small"],
  ["AP_m", "Medium"],
];
const PIPE: Record<string, string> = { fi1280: "Quick look", "fi1280+fi1920": "Closer look" };

export function Metrics() {
  const [m, setM] = useState<Record<string, any> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.metrics().then(setM).catch((e) => setErr(String(e.message)));
  }, []);

  if (err) return <main className="record"><p className="load-error">Couldn't load the evaluation reports: {err}</p></main>;
  if (!m) return <main className="record"><p className="note-pending">Opening the field record…</p></main>;

  const apRows: { label: string; ours?: Record<string, number>; theirs?: Record<string, number> }[] = [];
  const repro = m["repro_val"] ?? {};
  if (repro["fi1280/small=coco"]) apRows.push({ label: "Validation, quick look", ours: repro["fi1280/small=coco"], theirs: REPORT_AP["val fi1280"] });
  const two = Object.keys(repro).find((k) => k.startsWith("fi1280+fi1920/nms0.5"));
  if (two) apRows.push({ label: "Validation, closer look", ours: repro[two], theirs: REPORT_AP["val fi1280+fi1920"] });
  for (const p of ["fi1280", "fi1280+fi1920"]) {
    const t = m[`testdev_${p}`];
    if (t?.AP) apRows.push({ label: `Test-dev, ${PIPE[p].toLowerCase()}`, ours: t.AP, theirs: REPORT_AP[`testdev ${p}`] });
  }

  const cv = m["cv_fi1280"];
  const td = m["testdev_fi1280"];

  if (m["capstone"])
    return (
      <main className="record">
        <Guide
          g={m["capstone"]}
          methods={cv && <MethodBars cv={cv} />}
          testdev={td && <TestDev td={td} />}
          existence={cv && <ExistenceChart cv={cv} />}
          zeroshot={m["ovd_e1_val"] && <ZeroShot e1={m["ovd_e1_val"]} />}
          taught={m["ovd_e3_val"] && <Taught e3={m["ovd_e3_val"]} wise={m["ovd_e3_wise"]} />}
        />
      </main>
    );

  return (
    <main className="record">
      <section className="record-intro">
        <h1>How well it works</h1>
        <p>
          VisionQuery is only as good as its detector, so here is what the detector gets right and wrong. Every number below
          comes from VisDrone images the model never trained on. Tuning used the validation split with whole video
          sequences held out together; test-dev was scored once, after the settings were frozen.
        </p>
      </section>

      {apRows.length > 0 && (
        <section className="record-block">
          <h2>Finding objects</h2>
          <p className="record-lede">
            Average precision (AP) rewards finding every object with a tight box. The progress report's numbers are shown
            beside this re-run. Overall AP agrees to within 0.1, the gap you'd expect between a GPU at half precision and a
            CPU at full precision.
          </p>
          <div className="table-wrap">
            <table className="record-table">
              <thead>
                <tr>
                  <th scope="col">Split and detail</th>
                  {AP_COLS.map(([k, l]) => (
                    <th scope="col" key={k}>
                      {l}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {apRows.map((r) => (
                  <tr key={r.label}>
                    <th scope="row">{r.label}</th>
                    {AP_COLS.map(([k]) => (
                      <td key={k}>
                        <span className="strong">{r.ours?.[k]?.toFixed(1) ?? "–"}</span>
                        {r.theirs && <span className="was"> report {r.theirs[k].toFixed(1)}</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {cv && <ExistenceChart cv={cv} />}
      {cv && <MethodBars cv={cv} />}
      {td && <TestDev td={td} />}
      {m["ovd_e1_val"] && <ZeroShot e1={m["ovd_e1_val"]} />}
      {m["ovd_e3_val"] && <Taught e3={m["ovd_e3_val"]} wise={m["ovd_e3_wise"]} />}
    </main>
  );
}

/* Dumbbell: existence recall per class, counting threshold vs separate existence threshold. */
function ExistenceChart({ cv }: { cv: any }) {
  const rows = Object.entries(cv.existence_decision.per_class as Record<string, any>).sort((a, b) => a[1].rec_count_thr - b[1].rec_count_thr);
  const [hover, setHover] = useState<string | null>(null);
  const W = 640;
  const L = 170;
  const R = 24;
  const rowH = 30;
  const H = rows.length * rowH + 40;
  const lo = 0.5;
  const x = (v: number) => L + ((v - lo) / (1 - lo)) * (W - L - R);
  const d = cv.existence_decision;
  return (
    <section className="record-block">
      <h3>Noticing that something is there</h3>
      <p className="record-lede">
        For “is there a bus?” the app uses a lower, separate threshold than for counting. Measured with YOLO26s, the default model, on validation: for the four classes it used
        to miss most often, it now notices them {Math.round(d.weak_recall_separate * 100)}% of the time instead of {Math.round(d.weak_recall_count_thr * 100)}%,
        while its “yes” answers stay right {Math.round(d.pooled_prec_separate * 100)}% of the time (was {Math.round(d.pooled_prec_count_thr * 100)}%).
      </p>
      <ul className="chart-legend">
        <li>
          <i className="dot dot-before" /> Counting threshold
        </li>
        <li>
          <i className="dot dot-after" /> Separate presence threshold
        </li>
      </ul>
      <div className="chart-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Share of images where each class is noticed when present, before and after the separate presence threshold">
          {[0.5, 0.6, 0.7, 0.8, 0.9, 1].map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={8} y2={H - 26} className="grid" />
              <text x={x(t)} y={H - 8} className="tick" textAnchor="middle">
                {Math.round(t * 100)}%
              </text>
            </g>
          ))}
          {rows.map(([c, r], i) => {
            const y = 20 + i * rowH;
            const on = hover === c;
            return (
              <g key={c} onMouseEnter={() => setHover(c)} onMouseLeave={() => setHover(null)} className={on ? "row is-hover" : "row"}>
                <rect x={0} y={y - rowH / 2} width={W} height={rowH} className="hit" />
                <text x={L - 12} y={y + 4} className="row-label" textAnchor="end">
                  {FRIENDLY[c]}
                </text>
                <line x1={x(r.rec_count_thr)} x2={x(r.rec_separate)} y1={y} y2={y} className="dumbbell" />
                <circle cx={x(r.rec_count_thr)} cy={y} r={5} className="dot-before" />
                <circle cx={x(r.rec_separate)} cy={y} r={5} className="dot-after" />
                {on && (
                  <g className="tip">
                    <rect x={Math.min(x(r.rec_separate) + 12, W - 190)} y={y - 22} width={178} height={44} rx={3} />
                    <text x={Math.min(x(r.rec_separate) + 22, W - 180)} y={y - 5}>
                      noticed {Math.round(r.rec_count_thr * 100)}% → {Math.round(r.rec_separate * 100)}%
                    </text>
                    <text x={Math.min(x(r.rec_separate) + 22, W - 180)} y={y + 13}>
                      right when “yes” {Math.round(r.prec_separate * 100)}%
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}

function MethodBars({ cv }: { cv: any }) {
  const methods: [string, string, number][] = [
    ["A_per_class_thr", "One threshold per class (kept)", cv.A_per_class_thr.MAE],
    ["global_thr", "One threshold for all classes", cv.global_thr.MAE],
    ["D_density_thr", "Thresholds that shift with crowding", cv.D_density_thr?.MAE],
    ["B_expected_count", "Summed calibrated confidence", cv.B_expected_count.MAE],
    ["C_scaled_thr", "Per-class counts scaled for bias", cv.C_scaled_thr.MAE],
  ].filter((r) => r[2] != null) as [string, string, number][];
  const max = Math.max(...methods.map((m) => m[2]));
  return (
    <section className="record-block">
      <h3>Turning boxes into a count</h3>
      <p className="record-lede">
        Measured with YOLO26s on validation, five-fold and grouped by video: average miscount per image and class (lower is better). Four other ways of turning boxes into a
        count were tried; none beat a simple threshold per class, so that stays. Counting is limited by what the detector
        finds, not by how boxes are tallied.
      </p>
      <ul className="bars">
        {methods.map(([k, label, v]) => (
          <li key={k} className={k === "A_per_class_thr" ? "is-kept" : ""}>
            <span className="bars-label">{label}</span>
            <span className="bars-track">
              <span className="bars-fill" style={{ width: `${(v / max) * 100}%` }} />
            </span>
            <span className="bars-val">{v.toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function TestDev({ td }: { td: any }) {
  const b = td.bootstrap_per_class_minus_global;
  return (
    <section className="record-block">
      <h3>Counting and noticing, class by class</h3>
      <p className="record-lede">
        YOLO26s on VisDrone test-dev, scored once with every setting frozen. With the separate presence threshold, “is there a…?” is answered yes
        for {Math.round(td.per_class_thr_with_exist_thr.exist_rec * 100)}% of the class–image pairs that really contain the class (was{" "}
        {Math.round(td.per_class_thr.exist_rec * 100)}%), and those yeses are right {Math.round(td.per_class_thr_with_exist_thr.exist_prec * 100)}% of the time (was{" "}
        {Math.round(td.per_class_thr.exist_prec * 100)}%). Per-class counting thresholds miscount {Math.abs(b.sequence_unit[0]).toFixed(3)} less per cell than one
        shared threshold, but when whole video sequences are resampled the 95% interval runs from {b.sequence_unit[1].toFixed(3)} to{" "}
        {b.sequence_unit[2].toFixed(3)}, so that edge is not certain on this split.
      </p>
      <div className="table-wrap">
        <table className="record-table">
          <thead>
            <tr>
              <th scope="col">Class</th>
              <th scope="col">Objects</th>
              <th scope="col">Counted</th>
              <th scope="col">Miscount per image</th>
              <th scope="col">Noticed when present</th>
              <th scope="col">Right when “yes”</th>
            </tr>
          </thead>
          <tbody>
            {td.per_class.map((r: any) => (
              <tr key={r.class}>
                <th scope="row">{FRIENDLY[r.class]}</th>
                <td>{r.gt_boxes.toLocaleString()}</td>
                <td>{r.pred_boxes.toLocaleString()}</td>
                <td>{r.MAE.toFixed(2)}</td>
                <td>{Math.round(r.exist_rec * 100)}%</td>
                <td>{Math.round(r.exist_prec * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const SIZE_COLS: [string, string][] = [
  ["AP", "All"],
  ["AP_tiny", "Tiny"],
  ["AP_s", "Small"],
  ["AP_m", "Medium"],
  ["AP_l", "Large"],
];

function runLabel(name: string) {
  if (name.startsWith("yolo26s")) return "YOLO26s, trained on VisDrone";
  const [model, prompts] = name.split(":");
  const nice = model.replace("yoloe-26", "YOLOE-26").replace("yolo-world-v2", "YOLO-World v2").replace("owlv2", "OWLv2").replace("grounding-dino", "Grounding DINO");
  return `${nice}, ${prompts === "names" ? "class names" : "described"} prompts`;
}

/* Zero-shot prompt-based detectors vs the trained one (E1). */
function ZeroShot({ e1 }: { e1: any }) {
  const runs = Object.entries(e1.runs as Record<string, any>).sort((a, b) => (a[0].startsWith("yolo26s") ? -1 : b[0].startsWith("yolo26s") ? 1 : b[1].AP_all_sizes.AP - a[1].AP_all_sizes.AP));
  const best = e1.H1_best_zero_shot;
  const wordingRun = Object.entries(e1.runs as Record<string, any>).find(([, r]) => r.described_minus_names_ci_seq);
  const wording: number[] | undefined = wordingRun?.[1].described_minus_names_ci_seq;
  const gain = wordingRun ? wordingRun[1].AP_all_sizes.AP - (e1.runs[wordingRun[0].replace(":described", ":names")]?.AP_all_sizes.AP ?? 0) : 0;
  return (
    <section className="record-block">
      <h3>How the open-vocabulary detector compares</h3>
      <p className="record-lede">
        For things outside its ten classes the app asks an open-vocabulary detector, which finds objects from a text prompt
        without training on drone footage. On VisDrone's own classes it shows how much that costs:
        {best && <> the best one reaches {Math.round(best.fraction_of_finetuned * 100)}% of the trained detector's AP</>}, and it struggles most with
        tiny objects. That is why its answers are labelled as leads.
        {wording && (
          <>
            {" "}Wording matters: describing a class in plain English (“motorcycle” rather than VisDrone's “motor”) adds{" "}
            {gain.toFixed(1)} AP (95% interval {wording[0].toFixed(1)} to {wording[1].toFixed(1)}), so ask in everyday words.
          </>
        )}
      </p>
      <div className="table-wrap">
        <table className="record-table">
          <thead>
            <tr>
              <th scope="col">Detector</th>
              {SIZE_COLS.map(([k, l]) => (
                <th scope="col" key={k}>{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map(([name, r]) => (
              <tr key={name}>
                <th scope="row">{runLabel(name)}</th>
                {SIZE_COLS.map(([k]) => (
                  <td key={k}>
                    <span className={name.startsWith("yolo26s") ? "strong" : ""}>{r.AP_all_sizes[k].toFixed(1)}</span>
                    {k === "AP" && <span className="was">{r.AP_ci_seq[0].toFixed(1)}–{r.AP_ci_seq[1].toFixed(1)}</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* E3: YOLOE fine-tuned on SAM 3 labels. Learns the taught words, loses the rest. */
const E3_ROWS: [string, string][] = [
  ["visdrone", "VisDrone's 10 classes (vs people's labels)"],
  ["seen", "35 taught things (vs SAM 3)"],
  ["heldout", "8 words it was never taught (vs SAM 3)"],
];
const E3_MODEL: Record<string, string> = { zeroshot: "Zero-shot", tuned: "Drone-tuned" };
const SHOWCASE = ["tree", "building", "bush", "manhole cover", "street lamp", "crosswalk", "solar panel", "traffic light"];

function Taught({ e3, wise }: { e3: any; wise?: any }) {
  const g = e3.groups;
  const pc = g.seen.per_concept;
  const h = g.heldout.diff_vs_zeroshot?.tuned;
  const w = wise?.chosen;
  return (
    <section className="record-block">
      <h3>Teaching it drone things</h3>
      <p className="record-lede">
        The open-vocabulary detector was blind to trees and buildings, so SAM 3, a much bigger model, labelled drone frames
        overnight and YOLOE learned from those labels. It learned them well: trees went from {pc.zeroshot.tree.toFixed(0)} to{" "}
        {pc.tuned.tree.toFixed(0)} AP. But it forgot how to answer to words it wasn't taught
        {h && <> (−{Math.abs(h.diff).toFixed(1)} AP on the held-out words)</>}, even synonyms like “person” or “house”. So the app
        uses it only for the words it was taught and keeps the zero-shot model for everything else.
        {w && <> Blending the two models' weights ({Math.round(w.alpha * 100)}% tuned) keeps {w.keep_text}.</>}
      </p>
      <div className="table-wrap">
        <table className="record-table">
          <thead>
            <tr>
              <th scope="col">Asked about</th>
              {Object.keys(e3.models).map((k) => (
                <th scope="col" key={k}>{E3_MODEL[k] ?? k}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {E3_ROWS.map(([k, label]) => (
              <tr key={k}>
                <th scope="row">{label}</th>
                {Object.keys(e3.models).map((mk) => (
                  <td key={mk}>
                    <span className={mk === "tuned" ? "strong" : ""}>{g[k].AP[mk].toFixed(1)}</span>
                    {mk !== "zeroshot" && g[k].diff_vs_zeroshot?.[mk] && (
                      <span className="was">
                        {g[k].diff_vs_zeroshot[mk].ci95_seq[0] > 0 ? "+" : ""}
                        {g[k].diff_vs_zeroshot[mk].ci95_seq[0].toFixed(1)} to {g[k].diff_vs_zeroshot[mk].ci95_seq[1].toFixed(1)}
                      </span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-wrap">
        <table className="record-table">
          <thead>
            <tr>
              <th scope="col">Taught thing</th>
              <th scope="col">Zero-shot AP</th>
              <th scope="col">Drone-tuned AP</th>
            </tr>
          </thead>
          <tbody>
            {SHOWCASE.filter((c) => c in pc.tuned).map((c) => (
              <tr key={c}>
                <th scope="row">{c}</th>
                <td>{pc.zeroshot[c].toFixed(1)}</td>
                <td><span className="strong">{pc.tuned[c].toFixed(1)}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
