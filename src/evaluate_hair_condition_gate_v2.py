import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "hair_condition_gate_prepared_v2"
)

TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hair_condition_gate"
    / "hair_condition_gate_v2.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_ROOT
    / "models"
    / "hair_condition_gate"
    / "class_names.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "hair_condition_gate"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# START
# ============================================================

print("=" * 70)
print("HAIR CONDITION GATE V2 - EVALUATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_DIR}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file not found:\n{CLASS_NAMES_PATH}"
    )

print("All required files found.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_names = json.load(f)


print("\nClass names:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTest dataset loaded.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# GET TRUE LABELS
# ============================================================

print("\n" + "=" * 70)
print("COLLECTING PREDICTIONS")
print("=" * 70)

y_true = []

for images, labels in test_ds:

    y_true.extend(
        labels.numpy()
    )


y_true = np.array(y_true)


# ============================================================
# GET PREDICTIONS
# ============================================================

predictions = model.predict(
    test_ds,
    verbose=1
)


# ============================================================
# CONVERT PROBABILITY TO CLASS
# ============================================================

y_probability = predictions.flatten()

y_pred = (
    y_probability >= 0.5
).astype(int)


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


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    RESULTS_DIR
    / "classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "HAIR CONDITION GATE V2 - CLASSIFICATION REPORT\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Test Loss     : {test_loss:.4f}\n"
    )

    f.write(
        f"Test Accuracy : {test_accuracy:.4f}\n"
    )

    f.write(
        f"Test Accuracy : {test_accuracy * 100:.2f}%\n\n"
    )

    f.write(report)


print(
    f"\nClassification report saved to:\n{report_path}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\n")
print(cm)


# ============================================================
# PRINT CONFUSION MATRIX WITH LABELS
# ============================================================

print("\nConfusion Matrix Details:")

print(
    f"\nActual {class_names[0]}:"
)

print(
    f"  Predicted {class_names[0]} : {cm[0][0]}"
)

print(
    f"  Predicted {class_names[1]} : {cm[0][1]}"
)


print(
    f"\nActual {class_names[1]}:"
)

print(
    f"  Predicted {class_names[0]} : {cm[1][0]}"
)

print(
    f"  Predicted {class_names[1]} : {cm[1][1]}"
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = (
    RESULTS_DIR
    / "confusion_matrix.txt"
)


with open(
    cm_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "HAIR CONDITION GATE V2 - CONFUSION MATRIX\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"                {class_names[0]:<30}"
        f"{class_names[1]}\n"
    )

    f.write(
        f"Actual {class_names[0]:<22}"
        f"{cm[0][0]:<30}"
        f"{cm[0][1]}\n"
    )

    f.write(
        f"Actual {class_names[1]:<22}"
        f"{cm[1][0]:<30}"
        f"{cm[1][1]}\n"
    )


print(
    f"\nConfusion matrix saved to:\n{cm_path}"
)


# ============================================================
# CREATE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Hair Condition Gate V2 - Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=30,
    ha="right"
)

plt.yticks(
    tick_marks,
    class_names
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


# Add numbers inside matrix

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()


cm_image_path = (
    RESULTS_DIR
    / "confusion_matrix.png"
)


plt.savefig(
    cm_image_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nConfusion matrix image saved to:\n"
    f"{cm_image_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nTest images : {len(y_true)}"
)

print(
    f"Accuracy    : {test_accuracy * 100:.2f}%"
)

print(
    f"\nModel:\n{MODEL_PATH}"
)

print(
    f"\nClassification report:\n{report_path}"
)

print(
    f"\nConfusion matrix:\n{cm_image_path}"
)

print("\n" + "=" * 70)