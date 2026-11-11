"""YOLOE text-prompt trainer for datasets with different vocabularies.

Ultralytics pads every image's prompt list to the dataset's own class count,
then stacks the prompts of a batch into one tensor, so mixing a 45-name
dataset with a 10-name one crashes (or, worse, silently reshapes). It also pads
with the dataset's most frequent class names, which can duplicate a class that
is present in the image.

The YOLOE head also re-reads its prompt count from every forward pass, and its
loss is built after a dummy pass with the default 80 prompts, so the loss
expects 80 prompts per image whatever the vocabulary size (Ultralytics only
ever trains this path on 80+ class datasets, where the two agree).

This trainer pads every image to 80 prompts with *distractors*:
indoor and zoo things that a drone over a street essentially never sees
("piano", "giraffe"). They act as extra negatives and never collide with a
real label. The fix is re-applied when mosaic closes (Ultralytics rebuilds the
transforms then).
"""

from __future__ import annotations

from ultralytics.data import YOLOConcatDataset
from ultralytics.data.augment import RandomLoadText
from ultralytics.models.yolo.yoloe import YOLOETrainerFromScratch

DISTRACTORS = [
    "giraffe", "elephant", "zebra", "polar bear", "piano", "toothbrush", "hair drier", "toaster", "microwave oven",
    "refrigerator", "laptop", "computer keyboard", "computer mouse", "remote control", "teddy bear", "wine glass",
    "fork", "spoon", "sandwich", "broccoli", "hot dog", "pizza", "donut", "birthday cake", "bed", "toilet",
    "television", "kitchen sink", "snowboard", "skis", "surfboard", "violin", "grand piano", "harp", "saxophone",
    "accordion", "microscope", "stethoscope", "lipstick", "perfume bottle", "wristwatch", "necklace", "eyeglasses",
    "headphones", "chess piece", "playing card", "teapot", "frying pan", "cupcake", "sushi", "lobster", "octopus",
    "jellyfish", "penguin", "kangaroo", "camel", "rhinoceros", "hippopotamus", "flamingo", "lion", "tiger",
]
N_PROMPTS = 80  # what the loss expects, see above


class DroneOVTrainer(YOLOETrainerFromScratch):
    def _fix_text(self, dataset) -> None:
        n = N_PROMPTS
        for d in dataset.datasets if isinstance(dataset, YOLOConcatDataset) else [dataset]:
            for t in getattr(getattr(d, "transforms", None), "transforms", []):
                if isinstance(t, RandomLoadText):
                    t.max_samples, t.padding, t.padding_value = n, True, list(DISTRACTORS)

    def build_dataset(self, img_path, mode="train", batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            self._fix_text(ds)
        return ds

    def set_text_embeddings(self, datasets, batch):
        super().set_text_embeddings(datasets, batch)
        cache = next((d for d in datasets if hasattr(d, "category_names")), None)
        if cache is not None:
            from pathlib import Path

            self.text_embeddings.update(self.generate_text_embeddings(list(DISTRACTORS), batch, cache_dir=Path(cache.img_path).parent.parent))

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        self._fix_text(self.train_loader.dataset)
