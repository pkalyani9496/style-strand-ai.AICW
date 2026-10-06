from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "hair_condition_gate_prepared_v2"
)

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
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)

    return sorted(images)


random.seed(42)


# ============================================================
# SELECT IMAGES
# ============================================================

normal_train = get_images(
    DATASET_ROOT
    / "train"
    / "Normal_No_Visible_Condition"
)

disease_train = get_images(
    DATASET_ROOT
    / "train"
    / "Disease"
)

normal_val = get_images(
    DATASET_ROOT
    / "validation"
    / "Normal_No_Visible_Condition"
)

disease_val = get_images(
    DATASET_ROOT
    / "validation"
    / "Disease"
)

normal_test = get_images(
    DATASET_ROOT
    / "test"
    / "Normal_No_Visible_Condition"
)

disease_test = get_images(
    DATASET_ROOT
    / "test"
    / "Disease"
)


print("\n" + "=" * 70)
print("HAIR CONDITION GATE V2 INSPECTION")
print("=" * 70)

print("\nTRAIN")
print("Normal :", len(normal_train))
print("Disease:", len(disease_train))

print("\nVALIDATION")
print("Normal :", len(normal_val))
print("Disease:", len(disease_val))

print("\nTEST")
print("Normal :", len(normal_test))
print("Disease:", len(disease_test))


# ============================================================
# RANDOM SAMPLES
# ============================================================

normal_samples = random.sample(
    normal_train,
    min(5, len(normal_train))
)

disease_samples = random.sample(
    disease_train,
    min(5, len(disease_train))
)

normal_test_samples = random.sample(
    normal_test,
    min(5, len(normal_test))
)

disease_test_samples = random.sample(
    disease_test,
    min(5, len(disease_test))
)


samples = (
    [("NORMAL - TRAIN", x) for x in normal_samples]
    + [("DISEASE - TRAIN", x) for x in disease_samples]
    + [("NORMAL - TEST", x) for x in normal_test_samples]
    + [("DISEASE - TEST", x) for x in disease_test_samples]
)


# ============================================================
# DISPLAY
# ============================================================

fig, axes = plt.subplots(
    4,
    5,
    figsize=(18, 14)
)

axes = axes.flatten()

for i, (label, image_path) in enumerate(samples):

    try:

        img = Image.open(
            image_path
        ).convert("RGB")

        axes[i].imshow(img)

        axes[i].set_title(
            label + "\n" + image_path.name,
            fontsize=8
        )

        axes[i].axis("off")

    except Exception as e:

        print(
            f"Could not open {image_path}: {e}"
        )

        axes[i].axis("off")


fig.suptitle(
    "Hair Condition Gate V2 Dataset Inspection",
    fontsize=18
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "hair_condition_gate"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hair_condition_gate_v2_inspection.png"
)

plt.savefig(
    OUTPUT_FILE,
    dpi=150,
    bbox_inches="tight"
)

print("\nInspection image saved to:")
print(OUTPUT_FILE)

plt.show()