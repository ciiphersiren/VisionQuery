# Kaggle experiment kit

Three notebooks for the experiments that need a GPU. All are pre-registered in
the team's pre-registration notes (rounds 3 and 4); the decision rules there were
written before any open-vocabulary number was scored.

| Notebook | What it answers | GPU time (2×T4) | Send back |
|---|---|---|---|
| `e1_zero_shot.ipynb` | How well do zero-shot prompt-based detectors (YOLOE-26 s/m/l, YOLO-World-v2 s/l, OWLv2 base/large, Grounding DINO tiny/base) do on VisDrone compared with the fine-tuned YOLO26s, by object size, and does prompt wording matter? | ~2–3 h | `e1_results.zip` |
| `e2_heldout.ipynb` | Retrain YOLO26s without bus / bicycle / awning-tricycle, then measure how well the open-vocabulary branch covers the classes the trained model has never seen. | ~7 h | `e2_results.zip`, `yolo26s_heldout7.pt` |
| `e3b_frozen_alignment.ipynb` | E3 again with the text adapter and contrastive heads frozen (the layers that made it forget untaught words). Reuses E3's SAM 3 labels, so no HF token. | ~7.5 h | `yoloe-26s-drone-ov-b.pt`, `ovd_drone.json`, `e3b_results.zip` |
| `e3_drone_openvocab.ipynb` | **Overnight.** SAM 3 labels drone frames for 35 things VisDrone never labelled (trees, buildings, street lamps...); YOLOE-26 learns them with text prompts so any word still works; 8 held-out words test that. | ~11 h | `yoloe-26s-drone-ov.pt`, `ovd_drone.json`, `e3_results.zip` |

### E3 checklist (the overnight one)

1. Hugging Face: open huggingface.co/facebook/sam3 and accept the licence (SAM 3 is gated). Without it the
   notebook falls back to OWLv2-large, which is weaker.
2. Kaggle secret `HF_TOKEN` (read-only token), GPU T4 ×2, Internet on, `skyquery-code` dataset added
   (re-upload the zip, it contains the new `skyquery/ovtrain/` code).
3. Run it once with `QUICK = True` (≈20 min, interactive) and check every cell finishes.
4. Set `QUICK = False`, then **Save Version → Save & Run All (Commit)** and close the browser.
5. In the morning: from **Output** take `yoloe-26s-drone-ov.pt` → `skyquery/weights/`, `ovd_drone.json` →
   `skyquery/configs/`, and send me `e3_results.zip`. The app switches its open-vocabulary branch to the new
   weights on restart, but only if the pre-registered checks passed (`"adopt": true` in `ovd_drone.json`).

### E3 result (run 21 Oct 2026) and E3b

SAM 3 labelled all 548 val frames and 1,325 train frames in the 3 h budget (≈ 11 s per frame per T4). The tuned
YOLOE learned the taught words (tree 1 → 52 AP, building 2 → 48, VisDrone classes 11.6 → 31.9) but answers nothing
to untaught words (8.4 → 0.0), so `adopt` is false and the app routes only taught words to it. To run E3b: upload
your `e3_results.zip` as a dataset named `skyquery-e3-labels`, add it plus the new `skyquery-code`, GPU T4 ×2,
`QUICK = True` once, then Save & Run All. Send back `e3b_results.zip` and `yoloe-26s-drone-ov-b.pt`.

Knobs in the first cell: `MODEL_SIZE` (`"m"` is stronger but ~2× slower to train and to run on CPU),
`LABEL_HOURS` (more teacher time = more labelled frames, less training time), `TAU` (teacher score needed for a
box to count as a label).

## Steps

1. On Kaggle: **Datasets → New dataset**, upload `skyquery.zip` (the whole project), name it `skyquery-code`.
2. **Code → New notebook → File → Import notebook**, pick `e1_zero_shot.ipynb` (later `e2_heldout.ipynb`).
3. In the notebook: **Add data** → `skyquery-code`; **Settings** → Accelerator *GPU T4 ×2*, Internet *on*.
4. Optional, for OWLv2 and Grounding DINO: **Add-ons → Secrets** → add `HF_TOKEN` (a *read-only* token from
   huggingface.co/settings/tokens). The notebook reads it from Kaggle's secret store; never paste it into a cell.
5. **Save Version → Save & Run All (Commit)** so it runs without the browser open.
6. Download the zip(s) from the **Output** tab; the numbers go into the paper and the
   app's "How well it works" page.

If a model fails to download (for example Hugging Face is unreachable), E1 skips it and scores the rest.
E2 resumes training from `last.pt` if the session restarts.

## Running locally instead

Everything is plain Python; on any CUDA machine:

```bash
pip install -e ".[eval]" "transformers>=4.57" git+https://github.com/ultralytics/CLIP.git
python -m skyquery.calibrate.cache_ovd --split data/VisDrone2019-DET-val --model yoloe-26l --prompts names --device 0 --out cache/val_yoloe-26l_names.npz
python -m skyquery.calibrate.ovd_eval e1 --split data/VisDrone2019-DET-val --runs yoloe-26l:names=cache/val_yoloe-26l_names.npz
```

`python kaggle/make_notebooks.py` regenerates both notebooks from source.
