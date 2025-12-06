"""
Dataset class for Manga109s OCR training.

Loads text region crops from Manga109s dataset for training PaddleOCR-VL.
Supports optional data augmentation and synthetic data mixing.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset


load_dotenv()

MANGA109_ROOT = Path(os.getenv("MANGA109_ROOT", "")).expanduser()
DATA_SYNTHETIC_ROOT = Path(os.getenv("DATA_SYNTHETIC_ROOT", "")).expanduser()

PROMPT = "OCR:"


class MangaDataset(Dataset):
    """
    Dataset for Manga109s OCR training.
    
    Returns image-text pairs in the format expected by VL model processors.
    Preprocessing (tokenization, padding) is handled by the data collator.
    
    Args:
        split: Dataset split ('train' or 'test')
        limit_size: Limit dataset size (useful for evaluation)
        augment: Enable data augmentation (default False)
        skip_packages: Set of synthetic data package IDs to skip
        use_synthetic: Include synthetic data in training (default True)
    """

    def __init__(
        self,
        split: str,
        limit_size=None,
        augment: bool = False,
        skip_packages=None,
        use_synthetic: bool = True,
    ):
        data = []

        print(f"Initializing dataset {split}...")

        if skip_packages is None:
            skip_packages = set()
        else:
            skip_packages = {f"{x:04d}" for x in skip_packages}

        # Load synthetic data if available and enabled
        if use_synthetic and DATA_SYNTHETIC_ROOT.exists():
            meta_dir = DATA_SYNTHETIC_ROOT / "meta"
            if meta_dir.exists():
                for path in sorted(meta_dir.glob("*.csv")):
                    if path.stem in skip_packages:
                        continue
                    if not (DATA_SYNTHETIC_ROOT / "img" / path.stem).is_dir():
                        continue
                    df = pd.read_csv(path)
                    df = df.dropna()
                    df["path"] = df.id.apply(
                        lambda x, stem=path.stem: str(
                            DATA_SYNTHETIC_ROOT / "img" / stem / f"{x}.jpg"
                        )
                    )
                    df = df[["path", "text"]]
                    data.append(df)

        # Load Manga109 data
        data_csv = MANGA109_ROOT / "data.csv"
        if data_csv.exists():
            df = pd.read_csv(data_csv)
            df = df[df.split == split].reset_index(drop=True)
            df["path"] = df.crop_path.apply(lambda x: str(MANGA109_ROOT / x))
            df = df[["path", "text"]]
            data.append(df)
        else:
            raise FileNotFoundError(f"Dataset not found: {data_csv}")

        data = pd.concat(data, ignore_index=True)

        if limit_size:
            data = data.iloc[:limit_size]
        
        self.data = data
        self.augment = augment

        print(f"Dataset {split}: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return image and messages for VL model training.

        Returns:
            dict with 'images' (list of PIL Images) and 'messages' (chat format)
        """
        sample = self.data.loc[idx]
        text = sample.text
        image = Image.open(sample.path).convert("RGB")

        return {
            "images": [image],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                },
            ],
        }


if __name__ == "__main__":
    # Quick test
    ds = MangaDataset("train", limit_size=5)
    for i in range(min(5, len(ds))):
        sample = ds[i]
        print(f"Sample {i}: {sample['images'][0].size}, text: {sample['messages'][1]['content'][0]['text'][:50]}...")
