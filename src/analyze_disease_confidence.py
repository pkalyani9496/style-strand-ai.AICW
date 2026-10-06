import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hair_disease"
    / "hair_disease_mobilenetv2.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_ROOT
    / "models"
    / "hair_disease"
    / "class_names.json"
)

NORMAL_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "hair_type"
)

DISEASE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "hair_disease"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "hair_disease"
    / "confidence_analysis"
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

IMG_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("DISEASE CONFIDENCE ANALYSIS")
print("=" * 70)

print("\nLoading disease model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Disease model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("\nDisease classes:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# GET IMAGE FILES
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


normal_images = get_images(NORMAL_DIR)

disease_images = get_images(DISEASE_DIR)

print("\n" + "=" * 70)
print("IMAGE COUNTS")
print("=" * 70)

print(f"Normal candidate images: {len(normal_images)}")
print(f"Disease test images:     {len(disease_images)}")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    img = image.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Same MobileNetV2 preprocessing
    img_array = preprocess_input(
        img_array
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    max_confidence = float(
        np.max(predictions)
    )

    return (
        predicted_class,
        max_confidence,
        predictions
    )


# ============================================================
# ANALYZE NORMAL IMAGES
# ============================================================

print("\n" + "=" * 70)
print("ANALYZING NORMAL IMAGES")
print("=" * 70)

results = []

for index, image_path in enumerate(normal_images):

    predicted_class, confidence, probabilities = (
        predict_image(image_path)
    )

    results.append({
        "actual_group": "NORMAL",
        "image_path": str(image_path),
        "predicted_class": predicted_class,
        "max_confidence": confidence
    })

    if (index + 1) % 50 == 0:
        print(
            f"Processed normal images: "
            f"{index + 1}/{len(normal_images)}"
        )


# ============================================================
# ANALYZE DISEASE TEST IMAGES
# ============================================================

print("\n" + "=" * 70)
print("ANALYZING DISEASE TEST IMAGES")
print("=" * 70)

for index, image_path in enumerate(disease_images):

    predicted_class, confidence, probabilities = (
        predict_image(image_path)
    )

    # Determine actual disease class
    actual_class = image_path.parent.name

    results.append({
        "actual_group": "DISEASE",
        "actual_class": actual_class,
        "image_path": str(image_path),
        "predicted_class": predicted_class,
        "max_confidence": confidence
    })

    if (index + 1) % 100 == 0:
        print(
            f"Processed disease images: "
            f"{index + 1}/{len(disease_images)}"
        )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(results)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

csv_path = (
    OUTPUT_DIR
    / "confidence_analysis.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# STATISTICS
# ============================================================

normal_df = df[
    df["actual_group"] == "NORMAL"
]

disease_df = df[
    df["actual_group"] == "DISEASE"
]

print("\n" + "=" * 70)
print("NORMAL CONFIDENCE STATISTICS")
print("=" * 70)

print(
    normal_df["max_confidence"].describe()
)

print("\n" + "=" * 70)
print("DISEASE CONFIDENCE STATISTICS")
print("=" * 70)

print(
    disease_df["max_confidence"].describe()
)


# ============================================================
# IMPORTANT VALUES
# ============================================================

print("\n" + "=" * 70)
print("KEY CONFIDENCE VALUES")
print("=" * 70)

print("\nNORMAL:")
print(
    f"Minimum: {normal_df['max_confidence'].min():.4f}"
)

print(
    f"Maximum: {normal_df['max_confidence'].max():.4f}"
)

print(
    f"Mean:    {normal_df['max_confidence'].mean():.4f}"
)

print(
    f"Median:  {normal_df['max_confidence'].median():.4f}"
)


print("\nDISEASE:")
print(
    f"Minimum: {disease_df['max_confidence'].min():.4f}"
)

print(
    f"Maximum: {disease_df['max_confidence'].max():.4f}"
)

print(
    f"Mean:    {disease_df['max_confidence'].mean():.4f}"
)

print(
    f"Median:  {disease_df['max_confidence'].median():.4f}"
)


# ============================================================
# LOWEST-CONFIDENCE DISEASE IMAGES
# ============================================================

print("\n" + "=" * 70)
print("20 LOWEST-CONFIDENCE DISEASE IMAGES")
print("=" * 70)

lowest_disease = disease_df.sort_values(
    "max_confidence"
).head(20)

for _, row in lowest_disease.iterrows():

    print(
        f"{row['max_confidence']:.4f} | "
        f"{row['actual_class']} | "
        f"{row['predicted_class']} | "
        f"{Path(row['image_path']).name}"
    )


# ============================================================
# HIGHEST-CONFIDENCE NORMAL IMAGES
# ============================================================

print("\n" + "=" * 70)
print("20 HIGHEST-CONFIDENCE NORMAL IMAGES")
print("=" * 70)

highest_normal = normal_df.sort_values(
    "max_confidence",
    ascending=False
).head(20)

for _, row in highest_normal.iterrows():

    print(
        f"{row['max_confidence']:.4f} | "
        f"{row['predicted_class']} | "
        f"{Path(row['image_path']).name}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nResults saved to:")

print(csv_path)

print("\nDo NOT choose a threshold yet.")

print(
    "\nWe will inspect the confidence ranges first."
)