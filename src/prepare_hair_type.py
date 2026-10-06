from pathlib import Path
import random
import shutil
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

SOURCE_DIR = Path("datasets/hair_type")
OUTPUT_DIR = Path("datasets/hair_type_prepared")

CLASSES = [
    "Curly Hair",
    "Straight Hair",
    "Wavy Hair"
]

# Dataset split
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

# For reproducible splitting
RANDOM_SEED = 42

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# CHECK SOURCE DATASET
# ============================================================

print("=" * 70)
print("HAIR TYPE DATASET PREPARATION")
print("=" * 70)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Source dataset not found:\n{SOURCE_DIR.resolve()}"
    )

print(f"\nSource dataset: {SOURCE_DIR.resolve()}")
print(f"Output dataset: {OUTPUT_DIR.resolve()}")


# ============================================================
# REMOVE OLD PREPARED DATASET
# ============================================================

if OUTPUT_DIR.exists():

    print("\nRemoving previous prepared dataset...")

    shutil.rmtree(OUTPUT_DIR)

    print("Previous prepared dataset removed.")


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for split in ["train", "validation", "test"]:

    for class_name in CLASSES:

        class_dir = OUTPUT_DIR / split / class_name

        class_dir.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# PROCESS EACH CLASS
# ============================================================

total_original = 0
total_train = 0
total_validation = 0
total_test = 0


for class_name in CLASSES:

    print("\n" + "-" * 70)
    print(f"Processing: {class_name}")
    print("-" * 70)

    class_dir = SOURCE_DIR / class_name

    if not class_dir.exists():

        print(f"WARNING: Folder not found: {class_dir}")

        continue


    # --------------------------------------------------------
    # Find image files
    # --------------------------------------------------------

    image_files = [
        file
        for file in class_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
        and not file.name.startswith("._")
    ]


    # --------------------------------------------------------
    # Verify images
    # --------------------------------------------------------

    valid_images = []

    for image_path in image_files:

        try:

            with Image.open(image_path) as image:

                image.verify()

            valid_images.append(image_path)

        except Exception:

            print(f"Skipping corrupted image: {image_path.name}")


    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(valid_images)


    total_images = len(valid_images)

    total_original += total_images


    # --------------------------------------------------------
    # Calculate split sizes
    # --------------------------------------------------------

    train_count = int(
        total_images * TRAIN_RATIO
    )

    validation_count = int(
        total_images * VALIDATION_RATIO
    )

    test_count = (
        total_images
        - train_count
        - validation_count
    )


    # --------------------------------------------------------
    # Split images
    # --------------------------------------------------------

    train_images = valid_images[
        :train_count
    ]

    validation_images = valid_images[
        train_count:
        train_count + validation_count
    ]

    test_images = valid_images[
        train_count + validation_count:
    ]


    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    for image_path in train_images:

        destination = (
            OUTPUT_DIR
            / "train"
            / class_name
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination
        )


    for image_path in validation_images:

        destination = (
            OUTPUT_DIR
            / "validation"
            / class_name
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination
        )


    for image_path in test_images:

        destination = (
            OUTPUT_DIR
            / "test"
            / class_name
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination
        )


    # --------------------------------------------------------
    # Update totals
    # --------------------------------------------------------

    total_train += train_count
    total_validation += validation_count
    total_test += test_count


    # --------------------------------------------------------
    # Print class summary
    # --------------------------------------------------------

    print(f"Total images      : {total_images}")
    print(f"Training images   : {train_count}")
    print(f"Validation images : {validation_count}")
    print(f"Test images       : {test_count}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETED")
print("=" * 70)

print(f"\nTotal original images : {total_original}")
print(f"Training images       : {total_train}")
print(f"Validation images     : {total_validation}")
print(f"Test images           : {total_test}")

print("\nSplit percentages:")
print(f"Training   : {TRAIN_RATIO * 100:.0f}%")
print(f"Validation : {VALIDATION_RATIO * 100:.0f}%")
print(f"Test       : {TEST_RATIO * 100:.0f}%")

print("\nPrepared dataset:")
print(OUTPUT_DIR.resolve())

print("\nFolder structure:")
print("hair_type_prepared/")
print("├── train/")
print("│   ├── Curly Hair/")
print("│   ├── Straight Hair/")
print("│   └── Wavy Hair/")
print("├── validation/")
print("│   ├── Curly Hair/")
print("│   ├── Straight Hair/")
print("│   └── Wavy Hair/")
print("└── test/")
print("    ├── Curly Hair/")
print("    ├── Straight Hair/")
print("    └── Wavy Hair/")

print("\n" + "=" * 70)
print("READY FOR MODEL TRAINING")
print("=" * 70)