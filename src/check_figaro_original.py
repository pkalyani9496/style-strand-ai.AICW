from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGARO_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "hair_segmentation"
    / "Figaro1k"
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# FIND ORIGINAL TRAINING / TESTING FOLDERS
# ============================================================

training_dir = FIGARO_ROOT / "Figaro1k" / "Original" / "Training"
testing_dir = FIGARO_ROOT / "Figaro1k" / "Original" / "Testing"


# If the exact structure is different, search recursively.
if not training_dir.exists():

    possible_training = list(
        FIGARO_ROOT.rglob("Original/Training")
    )

    if possible_training:
        training_dir = possible_training[0]


if not testing_dir.exists():

    possible_testing = list(
        FIGARO_ROOT.rglob("Original/Testing")
    )

    if possible_testing:
        testing_dir = possible_testing[0]


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    images = []

    if not folder.exists():
        return images

    for path in folder.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)

    return images


training_images = get_images(training_dir)
testing_images = get_images(testing_dir)


print("\n" + "=" * 70)
print("FIGARO1K ORIGINAL DATASET INSPECTION")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nTraining folder:")
print(training_dir)

print("\nTesting folder:")
print(testing_dir)

print("\nTraining images:", len(training_images))
print("Testing images: ", len(testing_images))
print("Total images:   ", len(training_images) + len(testing_images))


# ============================================================
# SELECT RANDOM IMAGES
# ============================================================

random.seed(42)

training_sample_count = min(
    10,
    len(training_images)
)

testing_sample_count = min(
    10,
    len(testing_images)
)

training_sample = random.sample(
    training_images,
    training_sample_count
)

testing_sample = random.sample(
    testing_images,
    testing_sample_count
)


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

total_images = (
    training_sample_count
    + testing_sample_count
)

columns = 5
rows = (total_images + columns - 1) // columns

fig, axes = plt.subplots(
    rows,
    columns,
    figsize=(18, 4 * rows)
)

# Make axes always iterable
if total_images == 1:
    axes = [axes]
else:
    axes = axes.flatten()


# ============================================================
# TRAINING IMAGES
# ============================================================

position = 0

for image_path in training_sample:

    try:

        img = Image.open(
            image_path
        ).convert("RGB")

        axes[position].imshow(img)

        axes[position].set_title(
            "FIGARO TRAINING\n"
            + image_path.name,
            fontsize=8
        )

        axes[position].axis("off")

        position += 1

    except Exception as e:

        print(
            f"Could not open {image_path}: {e}"
        )


# ============================================================
# TESTING IMAGES
# ============================================================

for image_path in testing_sample:

    try:

        img = Image.open(
            image_path
        ).convert("RGB")

        axes[position].imshow(img)

        axes[position].set_title(
            "FIGARO TESTING\n"
            + image_path.name,
            fontsize=8
        )

        axes[position].axis("off")

        position += 1

    except Exception as e:

        print(
            f"Could not open {image_path}: {e}"
        )


# Hide unused axes

for i in range(position, len(axes)):
    axes[i].axis("off")


fig.suptitle(
    "Figaro1k Original Images Inspection",
    fontsize=18
)

plt.tight_layout()


# ============================================================
# SAVE RESULT
# ============================================================

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
    / "figaro_original_inspection.png"
)

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print("\nInspection image saved to:")
print(output_file)

plt.show()