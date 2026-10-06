import shutil
import random
from pathlib import Path

# ============================================================
# FIGARO1K HAIR SEGMENTATION DATASET PREPARATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGARO_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "hair_segmentation"
    / "Figaro1k"
    / "Figaro1k"
)

# ------------------------------------------------------------
# Source folders
# ------------------------------------------------------------

ORIGINAL_TRAIN = FIGARO_ROOT / "Original" / "Training"
ORIGINAL_TEST = FIGARO_ROOT / "Original" / "Testing"

GT_TRAIN = FIGARO_ROOT / "GT" / "Training"
GT_TEST = FIGARO_ROOT / "GT" / "Testing"

# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "hair_segmentation_prepared"
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

SEED = 42
TRAIN_RATIO = 0.80


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_image_files(folder):

    if not folder.exists():
        return []

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    files = []

    for file in folder.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() in valid_extensions:
            files.append(file)

    return sorted(files)


# ============================================================
# GET MASK FILES
# ============================================================

def get_mask_files(folder):

    if not folder.exists():
        return []

    masks = []

    for file in folder.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() != ".pbm":
            continue

        # Ignore duplicate files such as:
        # Frame01046-gt(1).pbm
        if "(1)" in file.stem:
            continue

        masks.append(file)

    return sorted(masks)


# ============================================================
# NORMALIZE FILENAMES
# ============================================================

def normalize_name(name):
    """
    Convert image and mask filenames to the same base name.

    Examples:

        Frame01046-org.jpg
        Frame01046-gt.pbm

    both become:

        frame01046
    """

    name = name.lower()

    # Remove extension
    name = Path(name).stem

    # Remove suffixes
    suffixes = [
        "-org",
        "_org",
        "-gt",
        "_gt",
        "-mask",
        "_mask",
        "-seg",
        "_seg"
    ]

    changed = True

    while changed:

        changed = False

        for suffix in suffixes:

            if name.endswith(suffix):

                name = name[:-len(suffix)]

                changed = True

                break

    # Remove duplicate marker
    name = name.replace("(1)", "")

    # Remove spaces
    name = name.replace(" ", "")

    return name


# ============================================================
# CREATE IMAGE-MASK PAIRS
# ============================================================

def create_pairs(image_folder, mask_folder):

    images = get_image_files(image_folder)
    masks = get_mask_files(mask_folder)

    print(f"Images found : {len(images)}")
    print(f"Masks found  : {len(masks)}")

    # --------------------------------------------------------
    # Create dictionary of masks
    # --------------------------------------------------------

    mask_dictionary = {}

    for mask in masks:

        key = normalize_name(mask.name)

        if key not in mask_dictionary:

            mask_dictionary[key] = mask

    # --------------------------------------------------------
    # Match images with masks
    # --------------------------------------------------------

    pairs = []
    missing = []

    for image in images:

        key = normalize_name(image.name)

        if key in mask_dictionary:

            mask = mask_dictionary[key]

            pairs.append(
                (image, mask)
            )

        else:

            missing.append(
                image.name
            )

    return pairs, missing


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

def create_output_folders():

    folders = [

        OUTPUT_ROOT / "train" / "images",
        OUTPUT_ROOT / "train" / "masks",

        OUTPUT_ROOT / "val" / "images",
        OUTPUT_ROOT / "val" / "masks",

        OUTPUT_ROOT / "test" / "images",
        OUTPUT_ROOT / "test" / "masks"

    ]

    for folder in folders:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY IMAGE + MASK
# ============================================================

def copy_pair(image, mask, split):

    image_destination = (
        OUTPUT_ROOT
        / split
        / "images"
        / image.name
    )

    mask_destination = (
        OUTPUT_ROOT
        / split
        / "masks"
        / mask.name
    )

    shutil.copy2(
        image,
        image_destination
    )

    shutil.copy2(
        mask,
        mask_destination
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 65)
    print("FIGARO1K HAIR SEGMENTATION DATASET PREPARATION")
    print("=" * 65)

    # ========================================================
    # CHECK SOURCE FOLDERS
    # ========================================================

    print("\nChecking source folders...\n")

    required_folders = [

        ORIGINAL_TRAIN,
        ORIGINAL_TEST,
        GT_TRAIN,
        GT_TEST

    ]

    for folder in required_folders:

        if folder.exists():

            print("FOUND :", folder)

        else:

            print("MISSING :", folder)

    for folder in required_folders:

        if not folder.exists():

            print("\nERROR: Required folder is missing.")

            print(folder)

            return

    # ========================================================
    # TRAINING IMAGE-MASK PAIRS
    # ========================================================

    print("\n")
    print("-" * 65)
    print("CHECKING TRAINING IMAGE-MASK PAIRS")
    print("-" * 65)

    train_pairs, missing_train = create_pairs(
        ORIGINAL_TRAIN,
        GT_TRAIN
    )

    print(
        "\nMatched training pairs :",
        len(train_pairs)
    )

    print(
        "Missing training masks :",
        len(missing_train)
    )

    # --------------------------------------------------------
    # If missing masks exist
    # --------------------------------------------------------

    if missing_train:

        print("\n")
        print("=" * 65)
        print("MISSING TRAINING MASKS")
        print("=" * 65)

        for filename in missing_train:

            print(
                " ",
                filename
            )

        print("\nDataset preparation stopped.")

        print(
            "No original files were deleted or modified."
        )

        return

    # ========================================================
    # TESTING IMAGE-MASK PAIRS
    # ========================================================

    print("\n")
    print("-" * 65)
    print("CHECKING TEST IMAGE-MASK PAIRS")
    print("-" * 65)

    test_pairs, missing_test = create_pairs(
        ORIGINAL_TEST,
        GT_TEST
    )

    print(
        "\nMatched testing pairs :",
        len(test_pairs)
    )

    print(
        "Missing testing masks :",
        len(missing_test)
    )

    # --------------------------------------------------------
    # If missing masks exist
    # --------------------------------------------------------

    if missing_test:

        print("\n")
        print("=" * 65)
        print("MISSING TESTING MASKS")
        print("=" * 65)

        for filename in missing_test:

            print(
                " ",
                filename
            )

        print("\nDataset preparation stopped.")

        print(
            "No original files were deleted or modified."
        )

        return

    # ========================================================
    # SHUFFLE TRAINING DATA
    # ========================================================

    random.seed(SEED)

    random.shuffle(
        train_pairs
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    train_count = int(
        len(train_pairs) * TRAIN_RATIO
    )

    train_split = train_pairs[
        :train_count
    ]

    val_split = train_pairs[
        train_count:
    ]

    # ========================================================
    # DISPLAY SPLIT
    # ========================================================

    print("\n")
    print("=" * 65)
    print("DATASET SPLIT")
    print("=" * 65)

    print(
        "\nTrain :",
        len(train_split),
        "pairs"
    )

    print(
        "Val   :",
        len(val_split),
        "pairs"
    )

    print(
        "Test  :",
        len(test_pairs),
        "pairs"
    )

    print(
        "Total :",
        len(train_split)
        + len(val_split)
        + len(test_pairs),
        "pairs"
    )

    # ========================================================
    # CREATE OUTPUT FOLDERS
    # ========================================================

    print("\nCreating output folders...")

    create_output_folders()

    # ========================================================
    # COPY TRAINING DATA
    # ========================================================

    print("\nCopying training pairs...")

    for index, (image, mask) in enumerate(
        train_split,
        start=1
    ):

        copy_pair(
            image,
            mask,
            "train"
        )

        if index % 100 == 0:

            print(
                f"  {index}/{len(train_split)}"
            )

    # ========================================================
    # COPY VALIDATION DATA
    # ========================================================

    print("\nCopying validation pairs...")

    for index, (image, mask) in enumerate(
        val_split,
        start=1
    ):

        copy_pair(
            image,
            mask,
            "val"
        )

        if index % 50 == 0:

            print(
                f"  {index}/{len(val_split)}"
            )

    # ========================================================
    # COPY TEST DATA
    # ========================================================

    print("\nCopying testing pairs...")

    for index, (image, mask) in enumerate(
        test_pairs,
        start=1
    ):

        copy_pair(
            image,
            mask,
            "test"
        )

        if index % 50 == 0:

            print(
                f"  {index}/{len(test_pairs)}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total = (
        len(train_split)
        + len(val_split)
        + len(test_pairs)
    )

    print("\n")
    print("=" * 65)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 65)

    print("\nFinal dataset:")

    print(
        "Train :",
        len(train_split),
        "pairs"
    )

    print(
        "Val   :",
        len(val_split),
        "pairs"
    )

    print(
        "Test  :",
        len(test_pairs),
        "pairs"
    )

    print(
        "Total :",
        total,
        "pairs"
    )

    print("\nOutput folder:")

    print(
        OUTPUT_ROOT
    )

    print("\nFolder structure:")

    print(
        """
hair_segmentation_prepared/
│
├── train/
│   ├── images/
│   └── masks/
│
├── val/
│   ├── images/
│   └── masks/
│
└── test/
    ├── images/
    └── masks/
"""
    )

    print("=" * 65)
    print("READY FOR SEGMENTATION MODEL TRAINING")
    print("=" * 65)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()