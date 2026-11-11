"""Command line: skyquery {ask,plan,detect,serve,eval}."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="visionquery", description="Ask questions about drone images.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="answer a question about an image")
    a.add_argument("image")
    a.add_argument("query")
    a.add_argument("--pipeline")
    a.add_argument("--model", help="model id from configs/models.json")
    a.add_argument("--save", help="write an overlay image here")
    a.add_argument("--json", action="store_true")

    p = sub.add_parser("plan", help="show the plan for a query (no image needed)")
    p.add_argument("query")

    d = sub.add_parser("detect", help="run the detector and print per-class counts")
    d.add_argument("image")
    d.add_argument("--pipeline")
    d.add_argument("--model", help="model id from configs/models.json")

    s = sub.add_parser("serve", help="run the API (and UI, if built)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    e = sub.add_parser("eval", help="AP + counting on cached predictions")
    e.add_argument("--split", required=True)
    e.add_argument("--cache", nargs="+", required=True, help="one npz per scale; several are fused")
    e.add_argument("--pipeline", default="fi1280")
    e.add_argument("--fusion-iou", type=float, default=0.6)

    args = ap.parse_args(argv)

    if args.cmd == "plan":
        from .planner.policy import Planner

        print(json.dumps(Planner()(args.query).model_dump(), indent=2))
        return

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("skyquery.api.server:app", host=args.host, port=args.port)
        return

    if args.cmd == "eval":
        from .calibrate.run import eval_cached

        print(json.dumps(eval_cached(args.split, args.cache, args.pipeline, args.fusion_iou), indent=2))
        return

    from .pipeline import SkyQuery, render

    sq = SkyQuery.from_config(pipeline=getattr(args, "pipeline", None), model=getattr(args, "model", None))
    if args.cmd == "detect":
        import numpy as np

        from .classes import CLASSES

        iid = sq.register(args.image)
        det = sq.detect(iid)
        cal = sq.calib(*sq.resolve(None, None))
        keep = det.conf >= cal.count_thr[det.cls]
        counts = np.bincount(det.cls[keep], minlength=len(CLASSES))
        print(f"{det.meta}")
        for c, n in zip(CLASSES, counts):
            print(f"  {c:16s} {n}")
        return

    r = sq.ask(args.image, args.query)
    if args.json:
        print(json.dumps({"plan": r.plan.model_dump(), "answer": r.answer.model_dump()}, indent=2))
    else:
        print(r.answer.text)
        for c in r.answer.caveats:
            print("  ⚠", c)
        print(f"  [plan: {r.plan.intent} {[t.classes for t in r.plan.targets]} region={r.plan.region} src={r.plan.source}; {r.detections.meta}]")
    if args.save:
        render(sq.image(r.image_id), r.answer, args.save)
        print(f"  overlay -> {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
