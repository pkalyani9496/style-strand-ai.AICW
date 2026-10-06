import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

TEST_DIR = "datasets/hair_presence_prepared_v2/test"

MODEL_PATH = (
    "models/hair_presence_v2/"
    "hair_presence_mobilenetv2_v2.keras"
)

CLASS_NAMES_PATH = (
    "models/hair_presence_v2/"
    "class_names.json"
)

RESULTS_DIR = "results/hair_presence_v2"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading V2 test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test dataset loaded.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("\nClass names:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Hair Presence V2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating V2 model...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)


print("\n" + "=" * 60)
print("HAIR PRESENCE V2 TEST RESULTS")
print("=" * 60)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# TRUE LABELS
# ============================================================

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_ds
])


# ============================================================
# PREDICTIONS
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
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\nPrediction Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(report_path, "w") as f:

    f.write(
        "HAIR PRESENCE V2 TEST RESULTS\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Test Loss     : {test_loss:.4f}\n"
    )

    f.write(
        f"Test Accuracy : "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write("=" * 60 + "\n")

    f.write(report)

    f.write("\n\nConfusion Matrix\n")

    f.write("=" * 60 + "\n")

    f.write(str(cm))


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "Hair Presence V2 - Confusion Matrix"
)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)


for i in range(len(class_names)):

    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(cm_path)

plt.close()


# ============================================================
# SAVE CONFUSION MATRIX CSV
# ============================================================

cm_csv_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.csv"
)

np.savetxt(
    cm_csv_path,
    cm,
    delimiter=",",
    fmt="%d"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("V2 EVALUATION COMPLETED")
print("=" * 60)

print("\nReport saved to:")
print(report_path)

print("\nConfusion matrix saved to:")
print(cm_path)

print("\nDone!")