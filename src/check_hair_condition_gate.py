import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "hair_condition_gate_prepared"
)

NORMAL_DIR = DATASET_DIR / "Normal_No_Visible_Condition"
DISEASE_DIR = DATASET_DIR / "Disease"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_images(folder):
    images = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return images


def load_random_images(folder, count=10):
    images = get_images(folder)

    if len(images) < count:
        count = len(images)

    return random.sample(images, count)


def create_contact_sheet():
    random.seed(42)

    normal_images = load_random_images(
        NORMAL_DIR,
        10
    )

    disease_images = load_random_images(
        DISEASE_DIR,
        10
    )

    fig, axes = plt.subplots(
        4,
        5,
        figsize=(18, 14)
    )

    axes = axes.flatten()

    # --------------------------------------------------------
    # Normal images
    # --------------------------------------------------------

    for i, image_path in enumerate(normal_images):

        image = Image.open(image_path).convert("RGB")

        axes[i].imshow(image)
        axes[i].set_title(
            f"NORMAL\n{image_path.name}",
            fontsize=8
        )
        axes[i].axis("off")

    # --------------------------------------------------------
    # Disease images
    # --------------------------------------------------------

    for i, image_path in enumerate(disease_images):

        position = 10 + i

        image = Image.open(image_path).convert("RGB")

        axes[position].imshow(image)
        axes[position].set_title(
            f"DISEASE\n{image_path.name}",
            fontsize=8
        )
        axes[position].axis("off")

    fig.suptitle(
        "Hair Condition Gate Dataset Inspection",
        fontsize=16
    )

    plt.tight_layout()

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "hair_condition_gate"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "dataset_inspection.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print("\nInspection image saved to:")
    print(output_file)


if __name__ == "__main__":
    create_contact_sheet()