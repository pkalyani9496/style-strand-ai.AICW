import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "hair_disease",
    "test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hair_disease",
    "hair_disease_mobilenetv2.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hair_disease",
    "class_names.json"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "hair_disease"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# 3. LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")


# ============================================================
# 4. LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"\nNumber of test images: {len(test_ds.file_paths)}")


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# 7. GET TRUE LABELS
# ============================================================

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])


# ============================================================
# 8. GET PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)


# Save report
report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("HAIR DISEASE CLASSIFICATION REPORT\n")
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

print(f"\nClassification report saved to:")
print(report_path)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 11. SAVE CONFUSION MATRIX AS CSV
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_csv_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(cm_csv_path)

print(f"\nConfusion matrix CSV saved to:")
print(cm_csv_path)


# ============================================================
# 12. PLOT CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 10)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues"
)

plt.title(
    "Hair Disease Classification - Confusion Matrix"
)

plt.tight_layout()

cm_image_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nConfusion matrix image saved to:")
print(cm_image_path)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("HAIR DISEASE EVALUATION COMPLETED")
print("=" * 60)

print("\nGenerated files:")

print("1.", report_path)
print("2.", cm_csv_path)
print("3.", cm_image_path)