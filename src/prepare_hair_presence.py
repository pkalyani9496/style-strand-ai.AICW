import os
import shutil
import random
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "datasets" / "hair_segmentation" / "Patch1k" / "Hair"

# Parent folder containing both Hair and NonHair
PATCH1K_DIR = BASE_DIR / "datasets" / "hair_segmentation" / "Patch1k"

HAIR_DIR = PATCH1K_DIR / "Hair"
NON_HAIR_DIR = PATCH1K_DIR / "NonHair"

OUTPUT_DIR = BASE_DIR / "datasets" / "hair_presence_prepared"


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# FIND IMAGES
# ============================================================

def get_images(folder):

    images = []

    if not folder.exists():
        return images

    for path in folder.rglob("*"):

        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return images


# ============================================================
# COPY DATASET
# ============================================================

def prepare_class(class_name, source_folder):

    print()
    print("=" * 70)
    print(f"PREPARING CLASS: {class_name}")
    print("=" * 70)

    images = get_images(source_folder)

    print(f"Images found: {len(images)}")

    if len(images) == 0:
        print(f"WARNING: No images found in:")
        print(source_folder)
        return

    # Shuffle images
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)

    validation_end = train_end + int(
        total * VALIDATION_RATIO
    )

    train_images = images[:train_end]

    validation_images = images[
        train_end:validation_end
    ]

    test_images = images[
        validation_end:
    ]

    splits = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images
    }

    # ========================================================
    # COPY EACH SPLIT
    # ========================================================

    for split_name, split_images in splits.items():

        destination = (
            OUTPUT_DIR
            / split_name
            / class_name
        )

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"{split_name}: "
            f"{len(split_images)} images"
        )

        for index, image_path in enumerate(
            split_images
        ):

            # Keep original filename
            destination_file = (
                destination
                / image_path.name
            )

            # Avoid duplicate filename problems
            if destination_file.exists():

                destination_file = (
                    destination
                    / f"{index}_{image_path.name}"
                )

            shutil.copy2(
                image_path,
                destination_file
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HAIR PRESENCE DATASET PREPARATION")
    print("=" * 70)

    print()
    print("Project directory:")
    print(BASE_DIR)

    print()
    print("Source directory:")
    print(PATCH1K_DIR)

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    # Set random seed
    random.seed(RANDOM_SEED)

    # ========================================================
    # CHECK SOURCE DIRECTORIES
    # ========================================================

    if not HAIR_DIR.exists():

        print()
        print("ERROR: Hair directory not found!")
        print(HAIR_DIR)
        return

    if not NON_HAIR_DIR.exists():

        print()
        print("ERROR: NonHair directory not found!")
        print(NON_HAIR_DIR)
        return

    # ========================================================
    # REMOVE OLD PREPARED DATASET
    # ========================================================

    if OUTPUT_DIR.exists():

        print()
        print("Removing old prepared dataset...")

        shutil.rmtree(
            OUTPUT_DIR
        )

    # ========================================================
    # CREATE DATASET
    # ========================================================

    prepare_class(
        "Hair",
        HAIR_DIR
    )

    prepare_class(
        "NonHair",
        NON_HAIR_DIR
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 70)

    print()

    for split in [
        "train",
        "validation",
        "test"
    ]:

        print(f"\n{split.upper()}")

        for class_name in [
            "Hair",
            "NonHair"
        ]:

            folder = (
                OUTPUT_DIR
                / split
                / class_name
            )

            count = len(
                get_images(folder)
            )

            print(
                f"{class_name}: {count}"
            )

    print()
    print("Prepared dataset location:")
    print(OUTPUT_DIR)

    print()
    print("Next step:")
    print(
        "Train the Hair / No-Hair classification model."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()