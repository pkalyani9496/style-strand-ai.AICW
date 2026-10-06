from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


# ============================================================
# SETTINGS
# ============================================================

TEST_DIR = Path("datasets/hair_type_prepared/test")

MODEL_PATH = Path(
    "models/hair_type/hair_type_mobilenetv2.keras"
)

CLASS_NAMES_PATH = Path(
    "models/hair_type/class_names.json"
)

RESULTS_DIR = Path("results/hair_type")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("HAIR TYPE MODEL EVALUATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR.resolve()}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH.resolve()}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file not found:\n"
        f"{CLASS_NAMES_PATH.resolve()}"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)


print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nTest directory:")
print(TEST_DIR.resolve())


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded successfully.")


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print("\nTest Loss:")
print(f"{test_loss:.4f}")

print("\nTest Accuracy:")
print(f"{test_accuracy:.4f}")

print(
    f"\nTest Accuracy Percentage: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# GET TRUE LABELS
# ============================================================

y_true = []

for images, labels in test_dataset:

    y_true.extend(
        labels.numpy()
    )


y_true = np.array(y_true)


# ============================================================
# GET PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

predictions = model.predict(
    test_dataset,
    verbose=1
)


y_pred = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# ACCURACY CHECK
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


print("\nAccuracy from predictions:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)


print("\n")
print(report)


# Save classification report

report_path = (
    RESULTS_DIR
    / "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "HAIR TYPE CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n")
print(cm)


# ============================================================
# DISPLAY CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

disp.plot(
    ax=ax)