
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import re

OUT = Path("data/plantvillage")

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading PlantVillage dataset...")

    ds = load_dataset(
        "geraldmc/plantvillage-full",
        revision="v0.1.0",
        split="train"
    )

    print("Total images:", len(ds))
    print("Exporting images...")

    counts = {"train": 0, "test": 0}

    for i, row in enumerate(tqdm(ds, desc="Saving images")):
        split = row["split"]
        label = safe_name(row["class_label"])

        class_dir = OUT / split / label
        class_dir.mkdir(parents=True, exist_ok=True)

        image = row["image"].convert("RGB")
        image.save(class_dir / f"{i:06d}.jpg", quality=95)

        counts[split] += 1

    print("\nDataset successfully exported!")
    print("Training images:", counts["train"])
    print("Testing images:", counts["test"])
    print("Location:", OUT.resolve())

if __name__ == "__main__":
    main()
