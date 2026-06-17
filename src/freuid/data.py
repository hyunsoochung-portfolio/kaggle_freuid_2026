"""Dataset + dataloaders.

STUB: the full dataset and its exact file/label layout land in June 2026. Fill in
``_index_samples`` once the sample data is on disk (``bash scripts/download_data.sh``).
Keep label convention consistent with metrics: 1 = fraud, 0 = bona-fide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Sample:
    path: Path
    label: int  # 1 = fraud, 0 = bona-fide; -1 = unknown (test)


class FreuidDataset(Dataset):
    """Image dataset yielding (tensor, label). Transforms passed in from the trainer."""

    def __init__(self, root: str | Path, split: str = "train", transform=None) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples: list[Sample] = self._index_samples()

    def _index_samples(self) -> list[Sample]:
        # TODO(team): implement once sample data layout is known (June 2026 release).
        # Expected shape, adjust to the real structure:
        #   data/train/bonafide/*.png   -> label 0
        #   data/train/fraud/*.png      -> label 1
        #   data/test/*.png             -> label -1
        raise NotImplementedError(
            "Dataset layout not wired up yet — inspect ./data after download_data.sh "
            "and implement _index_samples()."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label
