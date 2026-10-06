import os
import shutil
import json
import csv

from PIL import Image
import imagehash

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\CH KAVYA\OneDrive\Desktop\Hair Type Prediction"

# Original hair disease dataset
DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "hair_disease"
)

# Training images
TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

# Original test images
TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

# New clean test dataset
# IMPORTANT:
# This is a separate folder.
# The original test folder will NOT be modified.
CLEAN_TEST_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "hair_disease_clean_test"
)

# Existing trained model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hair_disease",
    "hair_disease_mobilenetv2.keras"
)

# Class names
CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hair_disease",
    "class_names.json"
)

# Results folder
RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "hair_disease",
    "clean_test_evaluation"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# Same threshold used in the previous near-duplicate check
THRESHOLD = 5

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_images(directory, split_name):

    images = []

    for root, dirs, files in os.walk(directory):

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            filepath = os.path.join(
                root,
                filename
            )

            # The class name is the immediate parent folder
            class_name = os.path.basename(root)

            images.append({
                "path": filepath,
                "filename": filename,
                "class": class_name,
                "split": split_name
            })

    return images


# ============================================================
# CALCULATE PERCEPTUAL HASH
# ============================================================

def calculate_hashes(images):

    print(
        f"\nCalculating hashes for "
        f"{len(images)} images..."
    )

    for i, image_info in enumerate(
        images,
        start=1
    ):

        try:

            with Image.open(
                image_info["path"]
            ) as img:

                img = img.convert("RGB")

                image_info["hash"] = (
                    imagehash.phash(img)
                )

        except Exception as e:

            image_info["hash"] = None

            print(
                f"\nCould not process:"
                f"\n{image_info['path']}"
                f"\nError: {e}"
            )

        if i % 500 == 0:

            print(
                f"Processed "
                f"{i}/{len(images)}"
            )


# ============================================================
# FIND TRAIN ↔ TEST NEAR DUPLICATES
# ============================================================

def find_contaminated_test_images(
    train_images,
    test_images
):

    # Test image paths that have at least one
    # near-duplicate in the training set
    contaminated_test_paths = set()

    # Store every matching train-test pair
    matches = []

    print(
        "\nComparing Train ↔ Test..."
    )

    for train_image in train_images:

        if train_image["hash"] is None:
            continue

        for test_image in test_images:

            if test_image["hash"] is None:
                continue

            distance = (
                train_image["hash"]
                -
                test_image["hash"]
            )

            if distance <= THRESHOLD:

                test_path = test_image["path"]

                contaminated_test_paths.add(
                    test_path
                )

                matches.append({
                    "distance": distance,
                    "train_path": train_image["path"],
                    "test_path": test_path,
                    "train_class": train_image["class"],
                    "test_class": test_image["class"],
                    "train_filename": train_image["filename"],
                    "test_filename": test_image["filename"]
                })

    return (
        contaminated_test_paths,
        matches
    )


# ============================================================
# SAVE MATCH INFORMATION
# ============================================================

def save_matches(matches):

    output_file = os.path.join(
        RESULTS_DIR,
        "train_test_near_duplicate_matches.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "TRAIN ↔ TEST NEAR-DUPLICATE MATCHES\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        for i, match in enumerate(
            matches,
            start=1
        ):

            f.write(
                f"Match {i}\n"
            )

            f.write(
                f"Distance : "
                f"{match['distance']}\n"
            )

            f.write(
                f"Train    : "
                f"{match['train_filename']}\n"
            )

            f.write(
                f"Train class : "
                f"{match['train_class']}\n"
            )

            f.write(
                f"Test     : "
                f"{match['test_filename']}\n"
            )

            f.write(
                f"Test class : "
                f"{match['test_class']}\n"
            )

            f.write(
                f"Train path : "
                f"{match['train_path']}\n"
            )

            f.write(
                f"Test path : "
                f"{match['test_path']}\n"
            )

            f.write(
                "\n"
                + "-" * 80
                + "\n\n"
            )

    print(
        "\nMatch information saved to:"
    )

    print(output_file)


# ============================================================
# CREATE CLEAN TEST DATASET
# ============================================================

def create_clean_test_dataset(
    test_images,
    contaminated_test_paths
):

    # Delete only our previously generated
    # clean-test folder if it exists.
    #
    # Original test dataset is NOT touched.
    if os.path.exists(CLEAN_TEST_DIR):

        print(
            "\nRemoving previous clean test folder..."
        )

        shutil.rmtree(
            CLEAN_TEST_DIR
        )

    os.makedirs(
        CLEAN_TEST_DIR,
        exist_ok=True
    )

    copied_count = 0
    removed_count = 0

    for image_info in test_images:

        source_path = image_info["path"]

        class_name = image_info["class"]

        destination_class_dir = os.path.join(
            CLEAN_TEST_DIR,
            class_name
        )

        os.makedirs(
            destination_class_dir,
            exist_ok=True
        )

        destination_path = os.path.join(
            destination_class_dir,
            image_info["filename"]
        )

        # Remove contaminated image from
        # the CLEAN copy only
        if source_path in contaminated_test_paths:

            removed_count += 1

            continue

        # Copy clean image
        shutil.copy2(
            source_path,
            destination_path
        )

        copied_count += 1

    return (
        copied_count,
        removed_count
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        class_names = json.load(f)

    return class_names


# ============================================================
# LOAD CLEAN TEST DATASET
# ============================================================

def load_clean_test_dataset():

    dataset = tf.keras.utils.image_dataset_from_directory(
        CLEAN_TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return dataset


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    dataset,
    class_names
):

    print(
        "\nLoading trained model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CLEAN TEST SET EVALUATION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Model evaluation
    # --------------------------------------------------------

    loss, accuracy = model.evaluate(
        dataset,
        verbose=1
    )

    print(
        f"\nClean Test Loss     : "
        f"{loss:.4f}"
    )

    print(
        f"Clean Test Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Clean Test Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    y_true = []
    y_pred = []

    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0
        )

        predicted_classes = (
            tf.argmax(
                predictions,
                axis=1
            ).numpy()
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            predicted_classes
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CLASSIFICATION REPORT"
    )

    print(
        "=" * 70
    )

    print(report)

    report_path = os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "=" * 70
    )

    print(cm)

    cm_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.csv"
    )

    with open(
        cm_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["Class"] + class_names
        )

        for class_name, row in zip(
            class_names,
            cm
        ):

            writer.writerow(
                [class_name] + list(row)
            )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)

print(
    "HAIR DISEASE CLEAN TEST SET CHECK"
)

print("=" * 70)


# ------------------------------------------------------------
# Check important paths
# ------------------------------------------------------------

print(
    "\nChecking required files and folders..."
)

if not os.path.exists(TRAIN_DIR):

    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(TEST_DIR):

    raise FileNotFoundError(
        f"Test folder not found:\n{TEST_DIR}"
    )

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(CLASS_NAMES_PATH):

    raise FileNotFoundError(
        f"Class names file not found:\n{CLASS_NAMES_PATH}"
    )

print(
    "All required paths found."
)


# ------------------------------------------------------------
# Load Train images
# ------------------------------------------------------------

print(
    "\nLoading Train images..."
)

train_images = collect_images(
    TRAIN_DIR,
    "train"
)

print(
    f"Train images: "
    f"{len(train_images)}"
)


# ------------------------------------------------------------
# Load Test images
# ------------------------------------------------------------

print(
    "\nLoading Test images..."
)

test_images = collect_images(
    TEST_DIR,
    "test"
)

print(
    f"Test images: "
    f"{len(test_images)}"
)


# ------------------------------------------------------------
# Calculate hashes
# ------------------------------------------------------------

calculate_hashes(
    train_images
)

calculate_hashes(
    test_images
)


# ------------------------------------------------------------
# Find contaminated test images
# ------------------------------------------------------------

(
    contaminated_test_paths,
    matches
) = find_contaminated_test_images(
    train_images,
    test_images
)


# ------------------------------------------------------------
# Print near-duplicate results
# ------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "NEAR-DUPLICATE ANALYSIS"
)

print(
    "=" * 70
)

print(
    f"\nNear-duplicate pairs : "
    f"{len(matches)}"
)

print(
    f"Unique Test images affected : "
    f"{len(contaminated_test_paths)}"
)


# ------------------------------------------------------------
# Save match details
# ------------------------------------------------------------

save_matches(
    matches
)


# ------------------------------------------------------------
# Create clean test dataset
# ------------------------------------------------------------

print(
    "\nCreating clean Test dataset..."
)

(
    copied_count,
    removed_count
) = create_clean_test_dataset(
    test_images,
    contaminated_test_paths
)

print(
    f"\nOriginal Test images : "
    f"{len(test_images)}"
)

print(
    f"Removed near-duplicates : "
    f"{removed_count}"
)

print(
    f"Clean Test images : "
    f"{copied_count}"
)


# ------------------------------------------------------------
# Load class names
# ------------------------------------------------------------

class_names = load_class_names()

print(
    "\nClasses:"
)

for i, class_name in enumerate(
    class_names
):

    print(
        f"{i}: {class_name}"
    )


# ------------------------------------------------------------
# Load clean test dataset
# ------------------------------------------------------------

print(
    "\nLoading clean Test dataset..."
)

clean_test_dataset = (
    load_clean_test_dataset()
)


# ------------------------------------------------------------
# Evaluate model
# ------------------------------------------------------------

evaluate_model(
    clean_test_dataset,
    class_names
)


# ------------------------------------------------------------
# Completed
# ------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "CLEAN TEST EVALUATION COMPLETED"
)

print(
    "=" * 70
)