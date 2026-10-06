from pathlib import Path
import random
import shutil


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# SOURCE DATASETS
# ============================================================

FIGARO_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "hair_segmentation"
    / "Figaro1k"
    / "Figaro1k"
    / "Original"
)

FIGARO_TRAIN = FIGARO_ROOT / "Training"
FIGARO_TEST = FIGARO_ROOT / "Testing"


DISEASE_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "hair_disease"
)

DISEASE_TRAIN = DISEASE_ROOT / "train"
DISEASE_TEST = DISEASE_ROOT / "test"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "hair_condition_gate_prepared_v2"
)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# PARAMETERS
# ============================================================

NORMAL_TRAIN_COUNT = 672
NORMAL_VAL_COUNT = 168
NORMAL_TEST_COUNT = 210

DISEASE_TRAIN_COUNT = 672
DISEASE_VAL_COUNT = 168
DISEASE_TEST_COUNT = 210

DISEASE_TRAIN_PER_CLASS = 84
DISEASE_TEST_PER_CLASS = 21


# ============================================================
# HELPER
# ============================================================

def get_images(folder):

    images = []

    for path in folder.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)

    return sorted(images)


def copy_images(
    images,
    destination,
    prefix
):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, source in enumerate(images):

        new_name = (
            f"{prefix}_{index:04d}"
            f"{source.suffix.lower()}"
        )

        destination_file = (
            destination / new_name
        )

        shutil.copy2(
            source,
            destination_file
        )


# ============================================================
# START
# ============================================================

print("\n" + "=" * 70)
print("HAIR CONDITION GATE V2 DATASET PREPARATION")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nOutput:")
print(OUTPUT_ROOT)


# ============================================================
# REMOVE OLD V2 DATASET IF EXISTS
# ============================================================

if OUTPUT_ROOT.exists():

    print(
        "\nExisting V2 dataset found."
    )

    print(
        "Removing old V2 dataset..."
    )

    shutil.rmtree(
        OUTPUT_ROOT
    )


# ============================================================
# LOAD FIGARO
# ============================================================

print("\n" + "=" * 70)
print("LOADING FIGARO NORMAL IMAGES")
print("=" * 70)

figaro_train_images = get_images(
    FIGARO_TRAIN
)

figaro_test_images = get_images(
    FIGARO_TEST
)

print(
    f"Figaro training images: "
    f"{len(figaro_train_images)}"
)

print(
    f"Figaro testing images:  "
    f"{len(figaro_test_images)}"
)


# ============================================================
# CHECK FIGARO COUNTS
# ============================================================

if len(figaro_train_images) != 840:

    raise ValueError(
        "Expected 840 Figaro training images."
    )

if len(figaro_test_images) != 210:

    raise ValueError(
        "Expected 210 Figaro testing images."
    )


# ============================================================
# SHUFFLE FIGARO TRAINING
# ============================================================

random.shuffle(
    figaro_train_images
)


normal_train = figaro_train_images[
    :NORMAL_TRAIN_COUNT
]

normal_val = figaro_train_images[
    NORMAL_TRAIN_COUNT:
    NORMAL_TRAIN_COUNT + NORMAL_VAL_COUNT
]

normal_test = figaro_test_images


print("\nNormal dataset:")
print(
    f"Train: {len(normal_train)}"
)

print(
    f"Validation: {len(normal_val)}"
)

print(
    f"Test: {len(normal_test)}"
)


# ============================================================
# COPY NORMAL IMAGES
# ============================================================

normal_train_dir = (
    OUTPUT_ROOT
    / "train"
    / "Normal_No_Visible_Condition"
)

normal_val_dir = (
    OUTPUT_ROOT
    / "validation"
    / "Normal_No_Visible_Condition"
)

normal_test_dir = (
    OUTPUT_ROOT
    / "test"
    / "Normal_No_Visible_Condition"
)


copy_images(
    normal_train,
    normal_train_dir,
    "normal"
)

copy_images(
    normal_val,
    normal_val_dir,
    "normal"
)

copy_images(
    normal_test,
    normal_test_dir,
    "normal"
)


# ============================================================
# LOAD DISEASE CLASSES
# ============================================================

print("\n" + "=" * 70)
print("LOADING DISEASE DATA")
print("=" * 70)

disease_classes = sorted(
    [
        folder.name
        for folder in DISEASE_TRAIN.iterdir()
        if folder.is_dir()
    ]
)

print(
    f"Disease classes found: "
    f"{len(disease_classes)}"
)

for class_name in disease_classes:

    print(
        f"  {class_name}"
    )


# ============================================================
# SELECT DISEASE TRAINING IMAGES
# ============================================================

print("\n" + "=" * 70)
print("SELECTING DISEASE TRAINING IMAGES")
print("=" * 70)

disease_train_selected = []

for class_name in disease_classes:

    class_dir = (
        DISEASE_TRAIN
        / class_name
    )

    images = get_images(
        class_dir
    )

    random.shuffle(
        images
    )

    selected = images[
        :DISEASE_TRAIN_PER_CLASS
    ]

    disease_train_selected.extend(
        selected
    )

    print(
        f"{class_name}: "
        f"{len(selected)} selected"
    )


print(
    f"\nTotal disease training images: "
    f"{len(disease_train_selected)}"
)


# ============================================================
# SPLIT DISEASE TRAINING INTO TRAIN / VALIDATION
# ============================================================

random.shuffle(
    disease_train_selected
)

disease_train = disease_train_selected[
    :DISEASE_TRAIN_COUNT
]

disease_val = disease_train_selected[
    DISEASE_TRAIN_COUNT:
]


print(
    f"Disease Train: "
    f"{len(disease_train)}"
)

print(
    f"Disease Validation: "
    f"{len(disease_val)}"
)


# ============================================================
# DISEASE TEST IMAGES
# ============================================================

print("\n" + "=" * 70)
print("SELECTING DISEASE TEST IMAGES")
print("=" * 70)

disease_test_selected = []

for class_name in disease_classes:

    class_dir = (
        DISEASE_TEST
        / class_name
    )

    images = get_images(
        class_dir
    )

    random.shuffle(
        images
    )

    selected = images[
        :DISEASE_TEST_PER_CLASS
    ]

    disease_test_selected.extend(
        selected
    )

    print(
        f"{class_name}: "
        f"{len(selected)} selected"
    )


print(
    f"\nTotal disease test images: "
    f"{len(disease_test_selected)}"
)


# ============================================================
# COPY DISEASE TRAIN
# ============================================================

disease_train_dir = (
    OUTPUT_ROOT
    / "train"
    / "Disease"
)

disease_val_dir = (
    OUTPUT_ROOT
    / "validation"
    / "Disease"
)

disease_test_dir = (
    OUTPUT_ROOT
    / "test"
    / "Disease"
)


copy_images(
    disease_train,
    disease_train_dir,
    "disease"
)

copy_images(
    disease_val,
    disease_val_dir,
    "disease"
)

copy_images(
    disease_test_selected,
    disease_test_dir,
    "disease"
)


# ============================================================
# FINAL COUNTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("\nTRAIN")
print(
    "Normal:",
    len(normal_train)
)

print(
    "Disease:",
    len(disease_train)
)

print(
    "Total:",
    len(normal_train) + len(disease_train)
)


print("\nVALIDATION")
print(
    "Normal:",
    len(normal_val)
)

print(
    "Disease:",
    len(disease_val)
)

print(
    "Total:",
    len(normal_val) + len(disease_val)
)


print("\nTEST")
print(
    "Normal:",
    len(normal_test)
)

print(
    "Disease:",
    len(disease_test_selected)
)

print(
    "Total:",
    len(normal_test)
    + len(disease_test_selected)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print("\nDataset created at:")

print(
    OUTPUT_ROOT
)

print("\nStructure:")

print(
    """
hair_condition_gate_prepared_v2/
│
├── train/
│   ├── Normal_No_Visible_Condition/
│   └── Disease/
│
├── validation/
│   ├── Normal_No_Visible_Condition/
│   └── Disease/
│
└── test/
    ├── Normal_No_Visible_Condition/
    └── Disease/
"""
)

print("=" * 70)