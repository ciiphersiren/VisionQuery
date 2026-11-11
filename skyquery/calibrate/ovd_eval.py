"""Scoring for the open-vocabulary experiments (E1 zero-shot gap, E2 held-out classes).

    python -m skyquery.calibrate.ovd_eval e1 --split data/VisDrone2019-DET-val \
        --runs yoloe-26s:names=cache/val_yoloe26s_names.npz yoloe-26s:described=cache/val_yoloe26s_described.npz

The fine-tuned reference is read from cache/<split>_1280.npz (FI1280).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..classes import CLASSES, CLASS_TO_ID
from ..config import ROOT
from ..eval.coco_eval import bootstrap_ap, coco_ap, eval_imgs, per_class_ap
from ..eval.counting import count_metrics, counts_at, prepare
from ..eval.visdrone import load_split, sequence_of
from .cache_preds import load_cache
from .tune import GRID, count_grid

REPORTS = ROOT / "reports"


def _r(d, n=2):
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


def subset(preds: dict, classes: list[str]) -> dict:
    ids = [CLASS_TO_ID[c] for c in classes]
    return {k: v[np.isin(v[:, 5], ids)] for k, v in preds.items()}


def existence_at_best_f1(preds_val, gts_val, preds_test, gts_test, classes):
    """Per class: threshold maximising image-level existence F1 on val, applied once to test."""
    out = {}
    _, Gv, Pv = prepare(preds_val, gts_val)
    _, Gt, Pt = prepare(preds_test, gts_test)
    CGv = count_grid(Pv)
    for c in classes:
        k = CLASS_TO_ID[c]
        pred = CGv[:, k, :] > 0
        gt = Gv[:, k] > 0
        tp = (pred & gt[:, None]).sum(0)
        prec = tp / np.maximum(pred.sum(0), 1)
        rec = tp / max(gt.sum(), 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        thr = float(GRID[int(np.argmax(f1))])
        th = np.full(len(CLASSES), 2.0)
        th[k] = thr
        C = counts_at(Pt, th)
        m = count_metrics(C[:, [k]], Gt[:, [k]])
        out[c] = {"threshold_from_val": thr, "test_exist_prec": round(m["exist_prec"], 3), "test_exist_rec": round(m["exist_rec"], 3), "test_MAE": round(m["MAE"], 3)}
    return out


def score_run(gts, preds, shapes, ref_E=None, n_boot=200):
    E, stems = eval_imgs(gts, preds, shapes)
    boot = bootstrap_ap(E, stems, n=n_boot, others=None)
    out = {"AP_all_sizes": _r(coco_ap(gts, preds, shapes)), "per_class_AP": _r(per_class_ap(gts, preds, shapes)), "AP_ci_seq": boot["ci"]}
    return out, E, stems


def cmd_e1(a):
    gts = load_split(a.split)
    tag = "testdev" if "test" in Path(a.split).name else "val"
    ref_preds, shapes = load_cache(ROOT / "cache" / f"{tag}_1280.npz")
    res = {"split": tag, "n_images": len(gts), "n_sequences": len({sequence_of(s) for s in gts}), "runs": {}}
    ref, E_ref, stems = score_run(gts, ref_preds, shapes, n_boot=a.boot)
    res["runs"]["yolo26s-finetuned (FI1280)"] = ref
    Es = {}
    for spec in a.runs:
        name, path = spec.split("=")
        preds, sh = load_cache(path)
        missing = set(gts) - set(preds)
        if missing:
            print(f"skip {name}: {len(missing)} images not cached yet")
            continue
        r, E, _ = score_run(gts, preds, {**shapes, **sh}, n_boot=a.boot)
        Es[name] = E
        b = bootstrap_ap(E_ref, stems, n=a.boot, others=[E])
        r["AP_minus_finetuned_ci_seq"] = b["diff_ci"][0]
        r["fraction_of_finetuned_AP"] = round(r["AP_all_sizes"]["AP"] / ref["AP_all_sizes"]["AP"], 3)
        res["runs"][name] = r
    # prompt-wording effect per model (H2): described - names
    for name in list(Es):
        if name.endswith(":names"):
            other = name.replace(":names", ":described")
            if other in Es:
                b = bootstrap_ap(Es[name], stems, n=a.boot, others=[Es[other]])
                res["runs"][other]["described_minus_names_ci_seq"] = b["diff_ci"][0]
    best = max((r["AP_all_sizes"]["AP"], n) for n, r in res["runs"].items() if n != "yolo26s-finetuned (FI1280)") if len(res["runs"]) > 1 else None
    if best:
        res["H1_best_zero_shot"] = {"run": best[1], "AP": best[0], "fraction_of_finetuned": round(best[0] / ref["AP_all_sizes"]["AP"], 3), "H1_supported (< 0.5)": best[0] / ref["AP_all_sizes"]["AP"] < 0.5}
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"ovd_e1_{tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: (v if k != "runs" else {n: {"AP": r["AP_all_sizes"]["AP"], "AP_tiny": r["AP_all_sizes"]["AP_tiny"], "ci": r["AP_ci_seq"], "vs_ft": r.get("AP_minus_finetuned_ci_seq"), "wording": r.get("described_minus_names_ci_seq")} for n, r in v.items()}) for k, v in res.items()}, indent=2))


def cmd_e2(a):
    """Held-out classes: oracle 10-class model vs 7-class model + YOLOE for the held-out ones."""
    held = a.holdout
    out = {"holdout": held}
    for tag, split in (("val", a.val), ("testdev", a.test)):
        gts = load_split(split)
        oracle, shapes = load_cache(a.oracle.format(split=tag))
        base7, _ = load_cache(a.base7.format(split=tag))
        ovd_held, _ = load_cache(a.ovd_held.format(split=tag))
        ovd_all, _ = load_cache(a.ovd_all.format(split=tag))
        runs = {
            "oracle (10-class fine-tuned)": oracle,
            "OVD prompted with held-out names": ovd_held,
            "OVD prompted with all ten names": subset(ovd_all, held),
        }
        block = {}
        for name, p in runs.items():
            pc = per_class_ap(gts, subset(p, held), shapes)
            block[name] = {c: round(pc[c], 2) for c in held}
            block[name]["mean"] = round(float(np.mean([pc[c] for c in held])), 2)
        hybrid = {k: np.concatenate([base7[k], ovd_held[k]]) for k in base7}
        block["hybrid AP, all 10 classes"] = round(coco_ap(gts, hybrid, shapes)["AP"], 2)
        block["oracle AP, all 10 classes"] = round(coco_ap(gts, oracle, shapes)["AP"], 2)
        base = [c for c in CLASSES if c not in held]
        pc7, pc10 = per_class_ap(gts, base7, shapes), per_class_ap(gts, oracle, shapes)
        block["base-class AP, 7-class model"] = round(float(np.mean([pc7[c] for c in base])), 2)
        block["base-class AP, 10-class model"] = round(float(np.mean([pc10[c] for c in base])), 2)
        out[tag] = block
        out[f"_{tag}_data"] = (gts, {"held": ovd_held, "all": subset(ovd_all, held), "oracle": oracle})
    gv, dv = out.pop("_val_data")
    gt, dt = out.pop("_testdev_data")
    out["existence_on_testdev"] = {name: existence_at_best_f1(dv[name], gv, dt[name], gt, held) for name in ("held", "all", "oracle")}
    ok = {}
    for name in ("held", "all"):
        e = out["existence_on_testdev"][name]
        ok[name] = sum(1 for c in held if e[c]["test_exist_prec"] >= 0.70 and e[c]["test_exist_rec"] >= 0.50)
    out["E2_claim_supported"] = max(ok.values()) >= 2
    out["E2_classes_passing"] = ok
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "ovd_e2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e1 = sub.add_parser("e1")
    e1.add_argument("--split", required=True)
    e1.add_argument("--runs", nargs="+", required=True, help="name=path.npz")
    e1.add_argument("--boot", type=int, default=200)
    e2 = sub.add_parser("e2")
    e2.add_argument("--val", required=True)
    e2.add_argument("--test", required=True)
    e2.add_argument("--holdout", nargs="+", default=["bus", "bicycle", "awning-tricycle"])
    e2.add_argument("--oracle", default=str(ROOT / "cache" / "{split}_1280.npz"))
    e2.add_argument("--base7", default=str(ROOT / "cache" / "{split}_heldout7_1280.npz"))
    e2.add_argument("--ovd-held", dest="ovd_held", default=str(ROOT / "cache" / "{split}_yoloe26l_heldout.npz"))
    e2.add_argument("--ovd-all", dest="ovd_all", default=str(ROOT / "cache" / "{split}_yoloe26l_names.npz"))
    a = ap.parse_args(argv)
    {"e1": cmd_e1, "e2": cmd_e2}[a.cmd](a)


if __name__ == "__main__":
    main()
