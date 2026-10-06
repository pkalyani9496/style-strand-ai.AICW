import os
import shutil
import random
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_SOURCE = PROJECT_ROOT / "datasets" / "hair_type"
DISEASE_SOURCE = PROJECT_ROOT / "datasets" / "hair_disease" / "train"

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "hair_condition_gate_prepared"

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# FUNCTIONS
# ============================================================

def get_images(folder):
    """Return all valid image files inside a folder recursively."""
    images = []

    if not folder.exists():
        return images

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return images


def split_files(files):
    """
    Split files into:
    Train = 80%
    Validation = 10%
    Test = 10%
    """

    files = list(files)

    random.shuffle(files)

    total = len(files)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    return train_files, val_files, test_files


def copy_files(files, destination):
    """Copy files into destination with unique filenames."""

    destination.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(files):
        new_name = f"{index:05d}_{source.name}"

        target = destination / new_name

        shutil.copy2(source, target)


def prepare_normal_class():
    """
    Use all 993 hair_type images as candidates for
    Normal_No_Visible_Condition.
    """

    print("\n" + "=" * 70)
    print("PREPARING NORMAL / NO VISIBLE CONDITION")
    print("=" * 70)

    normal_images = get_images(NORMAL_SOURCE)

    print(f"Normal source images found: {len(normal_images)}")

    if len(normal_images) == 0:
        raise RuntimeError(
            f"No images found in: {NORMAL_SOURCE}"
        )

    train, val, test = split_files(normal_images)

    print(f"Normal Train:      {len(train)}")
    print(f"Normal Validation: {len(val)}")
    print(f"Normal Test:       {len(test)}")

    normal_output = OUTPUT_DIR / "Normal_No_Visible_Condition"

    copy_files(train, normal_output / "train")
    copy_files(val, normal_output / "validation")
    copy_files(test, normal_output / "test")


def prepare_disease_class():
    """
    Select approximately 993 disease images from the
    existing disease TRAIN dataset.

    We select approximately equal numbers from all
    10 disease classes so that every condition is represented.
    """

    print("\n" + "=" * 70)
    print("PREPARING DISEASE CLASS")
    print("=" * 70)

    if not DISEASE_SOURCE.exists():
        raise RuntimeError(
            f"Disease source folder not found: {DISEASE_SOURCE}"
        )

    class_folders = [
        folder
        for folder in DISEASE_SOURCE.iterdir()
        if folder.is_dir()
    ]

    class_folders.sort()

    print(f"Disease classes found: {len(class_folders)}")

    for folder in class_folders:
        print(f"  {folder.name}")

    if len(class_folders) != 10:
        print(
            "\nWARNING: Expected 10 disease classes, "
            f"but found {len(class_folders)}."
        )

    # --------------------------------------------------------
    # Collect images from each disease class
    # --------------------------------------------------------

    class_images = {}

    for class_folder in class_folders:
        images = get_images(class_folder)

        if len(images) == 0:
            print(f"WARNING: No images in {class_folder.name}")
            continue

        class_images[class_folder.name] = images

        print(
            f"{class_folder.name}: {len(images)} images"
        )

    # --------------------------------------------------------
    # We need approximately 993 disease images.
    #
    # 993 / 10 = 99.3
    #
    # Therefore:
    # 3 classes → 100 images
    # 7 classes → 99 images
    # Total = 993
    # --------------------------------------------------------

    target_total = 993
    number_of_classes = len(class_images)

    if number_of_classes == 0:
        raise RuntimeError("No disease classes found.")

    base_count = target_total // number_of_classes
    extra_count = target_total % number_of_classes

    selected_images = []

    sorted_class_names = sorted(class_images.keys())

    print("\nSelecting disease images:")

    for index, class_name in enumerate(sorted_class_names):

        target_count = base_count

        if index < extra_count:
            target_count += 1

        available = class_images[class_name]

        if len(available) < target_count:
            raise RuntimeError(
                f"Class '{class_name}' has only "
                f"{len(available)} images, but "
                f"{target_count} are required."
            )

        random.shuffle(available)

        selected = available[:target_count]

        selected_images.extend(selected)

        print(
            f"{class_name}: selected {len(selected)}"
        )

    print(
        f"\nTotal selected disease images: "
        f"{len(selected_images)}"
    )

    # --------------------------------------------------------
    # Split selected disease images
    # --------------------------------------------------------

    train, val, test = split_files(selected_images)

    print(f"Disease Train:      {len(train)}")
    print(f"Disease Validation: {len(val)}")
    print(f"Disease Test:       {len(test)}")

    disease_output = OUTPUT_DIR / "Disease"

    copy_files(train, disease_output / "train")
    copy_files(val, disease_output / "validation")
    copy_files(test, disease_output / "test")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("HAIR CONDITION GATE DATASET PREPARATION")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nNormal source:")
    print(NORMAL_SOURCE)

    print(f"\nDisease source:")
    print(DISEASE_SOURCE)

    print(f"\nOutput:")
    print(OUTPUT_DIR)

    # Reproducibility
    random.seed(SEED)

    # --------------------------------------------------------
    # Remove old prepared dataset if it exists
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        print(
            "\nExisting prepared dataset found."
        )

        print(
            "Removing old prepared dataset..."
        )

        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Prepare both classes
    # --------------------------------------------------------

    prepare_normal_class()

    prepare_disease_class()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print("\nDataset created at:")

    print(OUTPUT_DIR)

    print("\nExpected structure:")

    print(
        """
datasets/
└── hair_condition_gate_prepared/
    ├── Normal_No_Visible_Condition/
    │   ├── train/
    │   ├── validation/
    │   └── test/
    │
    └── Disease/
        ├── train/
        ├── validation/
        └── test/
        """
    )


if __name__ == "__main__":
    main()