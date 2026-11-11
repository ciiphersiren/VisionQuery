/* "How it works": a walk through the whole project. Numbers come from reports/capstone.json
   and the other reports/*.json files; the sections passed in as props are drawn by Metrics.tsx. */
import type { ReactNode } from "react";

type Cov = { coverage: number | null; width: number | null; videos: number | null; infinite: boolean } | null;

const pct = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const f3 = (v: number | null | undefined) => (v == null ? "–" : v.toFixed(3));

function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div className="table-wrap">
      <table className="record-table">
        <thead>
          <tr>
            {head.map((h) => (
              <th scope="col" key={h}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (j === 0 ? <th scope="row" key={j}>{c}</th> : <td key={j}>{c}</td>))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function covCell(c: Cov) {
  if (!c) return "–";
  if (c.infinite) return "no finite range";
  return `${pct(c.coverage)} (width ${c.width})`;
}

interface Slots {
  g: any;
  methods?: ReactNode;
  testdev?: ReactNode;
  existence?: ReactNode;
  zeroshot?: ReactNode;
  taught?: ReactNode;
}

export function Guide({ g, methods, testdev, existence, zeroshot, taught }: Slots) {
  const d = g.detection as any[];
  const tile = d.find((r) => r.model === "YOLO26m-Tile");
  return (
    <>
      <section className="record-intro">
        <h1>How it works</h1>
        <p>
          This page walks through how VisionQuery answers a question, which models sit behind the model menu, how they were trained and tested,
          and how far each number can be trusted. Every figure comes from the project's result files, measured on images the models never
          trained on. Settings were chosen on validation splits only, and each test split was scored once.
        </p>
      </section>

      <section className="record-block">
        <h2>From a question to an answer</h2>
        <ol className="record-lede guide-steps">
          <li><strong>You ask</strong> in plain words: "how many cars are in the left half?"</li>
          <li><strong>The planner</strong> turns the words into a plan: what to do (count, check, find, compare, summarise), which object classes, and which of fifteen named parts of the photo. It follows VisDrone's own conventions: "pedestrian" is someone standing or walking, "people" someone sitting or riding, so a plain "people" means both. It never sees the photo and never produces a number; a language model is asked only when the rules recognise nothing.</li>
          <li><strong>The detector</strong> you picked in the model menu draws a box around every object it recognises. The boxes are kept, so follow-up questions about the same photo are instant.</li>
          <li><strong>The answer engine</strong> keeps the boxes of the right classes whose centres fall in the asked region, above a confidence threshold tuned for each class, and counts them.</li>
          <li><strong>The range</strong> beside a count says how far off it could be. For YOLO26m-Tile it is a 90% conformal interval (explained below).</li>
          <li><strong>Words it doesn't know</strong> go to an open-vocabulary detector. New things ("dog") are searched for directly; sub-types ("fire truck") keep only the detector's truck boxes that the open-vocabulary detector agrees with. Those answers are marked as leads, not counts.</li>
        </ol>
      </section>

      <section className="record-block">
        <h2>The models in the menu</h2>
        <p className="record-lede">
          Two drone datasets were used for training. VisDrone has ten classes (people, bikes, cars, vans, trucks, buses and more) in busy street scenes.
          UAVDT has drone video with only cars, trucks and buses. A third dataset, CARPK (parking lots), is used only for testing. The UAVDT models were
          chosen at their best validation epoch, which turned out to be the first, so they are the least trained.
        </p>
        <Table
          head={["Model", "Trained on", "Knows", "How it looks at a photo", "Score on its own test set (AP50)"]}
          rows={[
            ["YOLO26s (VisDrone)", "VisDrone", "10 classes", "whole photo at 1280 px, or 1280 + 1920", f3(d.find((r) => r.model === "YOLO26s")?.ap50) + " sliced"],
            ["YOLO26m-Tile (VisDrone)", "VisDrone", "10 classes", "512 px tiles enlarged 2.5× plus the whole photo", f3(tile?.ap50)],
            ["Ensemble", "VisDrone", "10 classes", "YOLO26m-Tile and YOLOE-26m-FT, boxes fused", f3(d.find((r) => r.model === "Ensemble")?.ap50)],
            ...(g.uavdt as any[]).filter((r) => r.split === "test").map((r) => [r.model, "UAVDT", "car, truck, bus", "whole photo at 1024 px", f3(r.ap50)]),
          ]}
        />
        <p className="record-lede">
          AP50 is average precision: the share of objects found with a box that overlaps the true box by at least half, traded off against false
          alarms; 1.0 is perfect. Scores on different test sets are not comparable with each other.
        </p>
      </section>

      <section className="record-block">
        <h2>Finding objects on VisDrone</h2>
        <p className="record-lede">
          All VisDrone models scored with the same scorer on the 1,610 test-dev images. Tiny objects (under 16 px) are the hardest; cutting the photo into
          enlarged tiles helps them most. Time is GPU time per image on one T4; on a laptop CPU the first question about a new photo takes seconds,
          follow-ups are instant.
        </p>
        <Table
          head={["Model", "AP50", "AP", "Tiny", "Small", "Medium+", "ms per image"]}
          rows={d.map((r) => [r.model, f3(r.ap50), f3(r.ap), f3(r.tiny), f3(r.small), f3(r.medium), r.ms])}
        />
        <p className="record-lede">Per class, for YOLO26m-Tile (AP50 on test-dev, with validation in brackets):</p>
        <Table head={["Class", "Objects in test-dev", "AP50"]} rows={(g.per_class as any[]).map((r) => [r.cls, r.objects.toLocaleString("en-US"), `${f3(r.ap50)} (${f3(r.val)})`])} />
      </section>

      <section className="record-block">
        <h2>Moving to other datasets</h2>
        <p className="record-lede">
          A model trained on one sky meets another. AP50 on the classes the datasets share (car, truck, bus; CARPK only cars). Parking-lot cars are big and
          tidy, so every model does well on CARPK; it says little about small objects.
        </p>
        <Table
          head={["Trained on", "VisDrone test-dev", "UAVDT test", "CARPK test"]}
          rows={(g.transfer as any[]).map((r) => [r.model, f3(r.visdrone), f3(r.uavdt), f3(r.carpk)])}
        />
      </section>

      <section className="record-block">
        <h2>Counting</h2>
        <p className="record-lede">
          A count is the number of boxes of the asked classes, in the asked region, above a confidence threshold. Each model has one threshold per
          class, tuned on its validation data to give the smallest counting error.
        </p>
        {methods}
        <h3>How close the counts are</h3>
        <p className="record-lede">Car counts per whole photo, against the true counts. Bias above zero means the model counts too many.</p>
        <Table
          head={["Where", "Photos", "Average cars", "Average error", "Bias"]}
          rows={(g.counting as any[]).map((r) => [r.label, r.images.toLocaleString("en-US"), r.mean, r.mae, r.bias > 0 ? `+${r.bias}` : r.bias])}
        />
        {testdev}
      </section>

      {existence && (
        <section className="record-block">
          <h2>Is there a …?</h2>
          {existence}
        </section>
      )}

      <section className="record-block">
        <h2>What the 90% range means</h2>
        <p className="record-lede">
          The range is built with conformal prediction: from past errors on validation photos, it picks a width that should contain the true count 9 times
          out of 10. Frames of one drone video are near-copies, so the width is calibrated on whole videos rather than single frames. Below, the share of
          test photos whose true count fell inside the range, and how often that held for a whole video.
        </p>
        <Table
          head={["Where", "Calibrated on videos", "Photos covered, whole videos", "Calibrated on frames"]}
          rows={(g.coverage as any[]).map((r) => [r.label, covCell(r.sequence), r.sequence?.videos == null || r.sequence.infinite ? "–" : pct(r.sequence.videos), covCell(r.image)])}
        />
        <p className="record-lede">By class on VisDrone (target 90%):</p>
        <Table head={["Class", "Calibrated on videos", "Calibrated on frames"]} rows={(g.classes as any[]).map((r) => [r.cls, covCell(r.sequence), covCell(r.image)])} />
        <p className="record-lede">Where it falls short, on VisDrone, before and after calibrating separately by predicted crowding or object size:</p>
        <Table head={["Scenes with", "Covered", "After adjusting"]} rows={(g.strata as any[]).map((r) => [r.label, pct(r.plain), pct(r.adjusted)])} />
        <ul className="record-lede">
          <li>On VisDrone the range holds for photos overall, but not for every video, and not for very crowded scenes or tiny objects.</li>
          <li>On UAVDT it is too narrow, because the VisDrone model over-counts there. Labelling up to 50 UAVDT frames does not fix it (best {pct(Math.max(...(g.few_shot as any[]).map((r) => r.uavdt)))}).</li>
          <li>On CARPK it is wider than needed, because the model counts parked cars very accurately.</li>
          <li>
            A simple Poisson range covers {pct(g.baselines.poisson)} of car counts on VisDrone at width {g.baselines.poisson_width}, about as well as the conformal range
            ({pct(g.baselines.conformal_car?.coverage)}, width {g.baselines.conformal_car?.width}), but it has no guarantee. The earlier "likely" range covered {pct(g.baselines.residual)} against its
            intended 80%.
          </li>
          <li>YOLO26s shows that "likely" range (the 10th to 90th percentile of its validation errors); the UAVDT models and the ensemble show no range yet.</li>
        </ul>
      </section>

      {(zeroshot || taught) && (
        <section className="record-block">
          <h2>Words outside the ten classes</h2>
          <p className="record-lede">
            The trained detectors only know VisDrone's ten classes (or three, for the UAVDT models). Anything else is sent to YOLOE, a detector that finds
            objects from a text prompt. It was not trained for drone footage, which is why its answers are labelled as leads.
          </p>
          {zeroshot}
          {taught}
        </section>
      )}

      <section className="record-block">
        <h2>Limits</h2>
        <ul className="record-lede">
          <li>Every model is a single training run.</li>
          <li>The question planner itself has not been tested on a set of real questions yet.</li>
          <li>UAVDT models rarely count buses: their bus threshold had to be set very high because the validation videos had few buses.</li>
          <li>The YOLO26s counting and "is there" results were measured with whole-image inference at 1280 px, so its detection score there differs from the sliced score in the model tables.</li>
          <li>The open-vocabulary answers are not calibrated for drone footage.</li>
        </ul>
      </section>
    </>
  );
}
