from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hair_type"
    / "hair_type_mobilenetv2.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_ROOT
    / "models"
    / "hair_type"
    / "class_names.json"
)


IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Hair Type model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Hair Type model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_names = json.load(f)


print("\nHair Type Classes:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# COLOUR RECOMMENDATIONS
# ============================================================

RECOMMENDATIONS = {

    "Curly Hair": [
        "Burgundy",
        "Copper",
        "Caramel Brown",
        "Chocolate Brown",
        "Purple"
    ],

    "Wavy Hair": [
        "Copper",
        "Caramel Brown",
        "Chocolate Brown",
        "Ash Brown",
        "Burgundy"
    ],

    "Straight Hair": [
        "Ash Brown",
        "Chocolate Brown",
        "Honey Blonde",
        "Burgundy",
        "Caramel Brown"
    ]
}


# ============================================================
# PREDICT HAIR TYPE
# ============================================================

def predict_hair_type(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


# ============================================================
# MAIN
# ============================================================

print("\n" + "=" * 70)
print("AI HAIR TYPE → COLOUR RECOMMENDATION")
print("=" * 70)

image_path = input(
    "\nEnter hair image path: "
).strip().strip('"')


if not Path(image_path).exists():

    print("\nERROR: Image file not found.")

else:

    hair_type, confidence, probabilities = (
        predict_hair_type(image_path)
    )

    print("\n" + "=" * 70)
    print("HAIR TYPE RESULT")
    print("=" * 70)

    print(
        f"\nDetected Hair Type : {hair_type}"
    )

    print(
        f"Confidence         : {confidence * 100:.2f}%"
    )

    print("\n" + "=" * 70)
    print("RECOMMENDED HAIR COLOURS")
    print("=" * 70)

    colors = RECOMMENDATIONS.get(
        hair_type,
        []
    )

    if not colors:

        print(
            "\nNo colour recommendations "
            "available for this hair type."
        )

    else:

        for i, color in enumerate(
            colors,
            start=1
        ):

            print(
                f"{i}. {color}"
            )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)