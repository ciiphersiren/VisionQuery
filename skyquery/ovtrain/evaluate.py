"""Score the zero-shot and the drone-tuned YOLOE side by side on VisDrone val.

Each model sees every val frame once, prompted with the whole vocabulary
(VisDrone descriptions + SEEN + HELDOUT), at 1280 px. Three scores:

  visdrone  AP on the 10 classes against the human labels (ignored regions as crowd)
  seen      AP on the taught extra concepts against the teacher's labels (score >= tau)
  heldout   AP on concepts neither model was taught, against the teacher's labels

"heldout" is the open-vocabulary test: if fine-tuning had destroyed the
ability to read new prompts, it would drop below zero-shot. Every difference
comes with a 95% interval from resampling whole video sequences (frames of one
sequence are near-duplicates). The teacher's labels are not ground truth, so
"seen" and "heldout" measure agreement with SAM 3, a strong but imperfect
reference; say so when reporting them.

Also picks the single count threshold with the lowest MAE on SEEN (for the
app's ovd_threshold) and reports it on HELDOUT.

    python -m skyquery.ovtrain.evaluate --val data/VisDrone2019-DET-val --teacher-val cache/teacher_val_*.npz \
        --model base=yoloe-26s-seg.pt --model tuned=weights/yoloe-26s-drone.pt --device 0 --out reports/ovd_e3_val.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..eval.coco_eval import bootstrap_ap, eval_imgs, per_class_ap
from ..eval.visdrone import ImageGT, load_split
from . import vocab
from .build import load_cache

GROUPS = {"visdrone": vocab.VISDRONE, "seen": vocab.SEEN, "heldout": vocab.HELDOUT}


def seq_half(stem: str) -> str:
    """Deterministic split of val into two halves by whole sequences (for choosing a setting on one, reporting on the other)."""
    import hashlib

    from ..eval.visdrone import sequence_of

    return "A" if int(hashlib.md5(sequence_of(stem).encode()).hexdigest(), 16) % 2 == 0 else "B"


def all_prompts() -> list[str]:
    return vocab.teacher_prompts(vocab.VISDRONE + vocab.SEEN + vocab.HELDOUT)


def offsets() -> dict[str, tuple[int, int]]:
    out, k = {}, 0
    for g, names in GROUPS.items():
        out[g] = (k, k + len(names))
        k += len(names)
    return out


def predict_all(weights: str, images: list[Path], prompts: list[str], imgsz: int, device: str, cache: Path) -> dict[str, np.ndarray]:
    done: dict[str, np.ndarray] = {}
    if cache.exists():
        with np.load(cache) as z:
            done = {k: z[k] for k in z.files}
    todo = [p for p in images if p.stem not in done]
    if todo:
        from ultralytics import YOLOE

        m = YOLOE(weights)
        m.set_classes(prompts, m.get_text_pe(prompts))
        t0 = time.time()
        for i, p in enumerate(todo, 1):
            r = m.predict(str(p), imgsz=imgsz, conf=0.005, max_det=1000, device=device, verbose=False)[0]
            b = r.boxes
            done[p.stem] = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1).astype(np.float32)
            if i % 50 == 0 or i == len(todo):
                np.savez_compressed(cache, **done)
                print(f"[eval {cache.stem}] {i}/{len(todo)} {(time.time() - t0) / i:.2f} s/frame", flush=True)
    return done


def teacher_gt(tv: dict[str, np.ndarray], meta: dict, names: list[str], tau: float) -> dict[str, ImageGT]:
    idx = {p: i for i, p in enumerate(meta.get("prompts", []))}
    want = {idx[vocab.first(n)]: j for j, n in enumerate(names) if vocab.first(n) in idx}
    out = {}
    for stem, a in tv.items():
        a = a[(a[:, 4] >= tau) & np.isin(a[:, 5].astype(int), list(want))]
        out[stem] = ImageGT(stem, a[:, :4].astype(np.float32), np.array([want[int(c)] for c in a[:, 5]], np.int64), np.zeros((0, 4), np.float32))
    return out


def group_preds(preds: dict[str, np.ndarray], lo: int, hi: int, stems) -> dict[str, np.ndarray]:
    out = {}
    for s in stems:
        a = preds.get(s, np.zeros((0, 6), np.float32))
        a = a[(a[:, 5] >= lo) & (a[:, 5] < hi)].copy()
        a[:, 5] -= lo
        out[s] = a
    return out


def count_mae(gts: dict[str, ImageGT], preds: dict[str, np.ndarray], k: int, thr: float) -> float:
    errs = []
    for s, g in gts.items():
        gt = np.bincount(g.cls, minlength=k)
        p = preds[s]
        pc = np.bincount(p[p[:, 4] >= thr, 5].astype(int), minlength=k)
        errs.append(np.abs(gt - pc))
    return float(np.mean(errs)) if errs else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--teacher-val", nargs="+", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=weights, first one is the baseline")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--half", choices=["all", "A", "B"], default="all", help="score one half of the val sequences (A/B split by sequence-name hash)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-config", default="", help="also write the app's configs/ovd_drone.json here")
    ap.add_argument("--weights-name", default="", help="weights path to put in that file (relative to the project)")
    a = ap.parse_args(argv)

    from PIL import Image

    models = dict(m.split("=", 1) for m in a.model)
    gts_vd = load_split(a.val)
    tv, _, meta = load_cache(a.teacher_val)
    stems = sorted(s for s in gts_vd if s in tv)[: a.limit or None]  # frames the teacher labelled
    if a.half != "all":
        stems = [s for s in stems if seq_half(s) == a.half]
    if not stems:
        raise SystemExit("no val frame has teacher labels")
    images = [Path(a.val) / "images" / f"{s}.jpg" for s in stems]
    shapes = {s: Image.open(p).size for s, p in zip(stems, images)}
    prompts = all_prompts()
    off = offsets()
    refs = {
        "visdrone": {s: gts_vd[s] for s in stems},
        "seen": teacher_gt({s: tv[s] for s in stems}, meta, vocab.SEEN, a.tau),
        "heldout": teacher_gt({s: tv[s] for s in stems}, meta, vocab.HELDOUT, a.tau),
    }
    preds = {}
    for name, w in models.items():
        preds[name] = predict_all(w, images, prompts, a.imgsz, a.device, Path(a.cache_dir) / f"e3_{name}_{a.imgsz}.npz")

    res = {"frames": len(stems), "half": a.half, "teacher": meta.get("teacher"), "tau": a.tau, "imgsz": a.imgsz, "models": models, "groups": {}}
    base = next(iter(models))
    for g, names in GROUPS.items():
        lo, hi = off[g]
        labels = vocab.teacher_prompts(names)
        gp = {m: group_preds(preds[m], lo, hi, stems) for m in models}
        E = {m: eval_imgs(refs[g], gp[m], shapes, names=labels)[0] for m in models}
        _, order = eval_imgs(refs[g], gp[base], shapes, names=labels)
        others = [m for m in models if m != base]
        b = bootstrap_ap(E[base], order, n=a.boot, others=[E[m] for m in others])
        out = {"reference": "human labels" if g == "visdrone" else f"{meta.get('teacher')} labels, score >= {a.tau}",
               "instances": int(sum(len(x.cls) for x in refs[g].values())),
               "AP": {base: b["ap"]}, "AP_ci_seq": {base: b["ci"]}, "diff_vs_" + base: {}, "per_concept": {}}
        for m, d in zip(others, b["diff_ci"]):
            out["AP"][m] = bootstrap_ap(E[m], order, n=1)["ap"]
            out["diff_vs_" + base][m] = {"diff": round(out["AP"][m] - out["AP"][base], 2), "ci95_seq": d}
        for m in models:
            out["per_concept"][m] = {k: round(v, 2) for k, v in per_class_ap(refs[g], gp[m], shapes, names=labels).items()}
        if g != "visdrone":
            thr = np.round(np.arange(0.05, 0.61, 0.05), 2)
            out["count_mae"] = {m: {str(t): round(count_mae(refs[g], gp[m], len(names), t), 3) for t in thr} for m in models}
        res["groups"][g] = out

    seen_mae = res["groups"]["seen"]["count_mae"]
    res["app_threshold"] = {m: float(min(v, key=v.get)) for m, v in seen_mae.items()}
    res["app_threshold_heldout_mae"] = {m: res["groups"]["heldout"]["count_mae"][m][str(t)] for m, t in res["app_threshold"].items()}

    # decision rule fixed before the run, for the second model vs the first
    if len(models) > 1:
        tuned = list(models)[1]
        d = {g: res["groups"][g]["diff_vs_" + base][tuned] for g in GROUPS}
        res["decision"] = {
            "H1_visdrone_better": d["visdrone"]["ci95_seq"][0] > 0,
            "H2_seen_better": d["seen"]["ci95_seq"][0] > 0,
            "H3_open_vocab_preserved": d["heldout"]["ci95_seq"][0] > -1.0,
            "H3_unseen_words_improve": d["heldout"]["ci95_seq"][0] > 0,
        }
        res["decision"]["adopt_in_app"] = res["decision"]["H1_visdrone_better"] and res["decision"]["H3_open_vocab_preserved"]
        if a.write_config:
            Path(a.write_config).write_text(json.dumps({
                "weights": a.weights_name or models[tuned], "threshold": res["app_threshold"][tuned],
                "adopt": res["decision"]["adopt_in_app"], "source": Path(a.out).name, "teacher": meta.get("teacher"),
                # not adopted for every word, but clearly better on the taught ones: use it for those only
                "route_taught": (not res["decision"]["adopt_in_app"]) and res["decision"]["H2_seen_better"],
            }, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))

    print(f"\n{len(stems)} val frames, teacher={meta.get('teacher')}")
    print(f"{'':10s}" + "".join(f"{m:>12s}" for m in models) + "   diff (95% CI, sequence bootstrap)")
    for g, o in res["groups"].items():
        diffs = "  ".join(f"{m}: {d['diff']:+.2f} [{d['ci95_seq'][0]:+.2f}, {d['ci95_seq'][1]:+.2f}]" for m, d in o["diff_vs_" + base].items())
        print(f"{g:10s}" + "".join(f"{o['AP'][m]:12.2f}" for m in models) + "   " + diffs)
    for c in ("tree", "building"):
        print(f"  AP[{c}]: " + "  ".join(f"{m} {res['groups']['seen']['per_concept'][m].get(c, float('nan')):.2f}" for m in models))
    print("count threshold for the app:", res["app_threshold"])
    if "decision" in res:
        print("pre-registered decision:", res["decision"])


if __name__ == "__main__":
    main()
