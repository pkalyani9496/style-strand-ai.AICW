from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hair_condition_gate"
    / "hair_condition_gate_v2.keras"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("HAIR CONDITION GATE - REAL IMAGE TEST")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# GET IMAGE PATH
# ============================================================

image_path = input(
    "\nEnter the FULL path of your hair image:\n"
).strip()

# Remove quotation marks if pasted with them
image_path = image_path.strip('"').strip("'")


# ============================================================
# CHECK IMAGE
# ============================================================

image_file = Path(image_path)

if not image_file.exists():
    print("\nERROR: Image file not found.")
    print("Please check the image path.")
    raise SystemExit


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image...")

try:
    image = Image.open(image_file).convert("RGB")

except Exception as e:
    print(f"\nERROR: Could not open image.\n{e}")
    raise SystemExit


print(f"Original image size: {image.size}")


# ============================================================
# RESIZE
# ============================================================

image = image.resize(IMAGE_SIZE)


# ============================================================
# CONVERT IMAGE TO ARRAY
# ============================================================

image_array = np.array(
    image,
    dtype=np.float32
)

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# PREDICTION
# ============================================================

print("\nAnalyzing image...")

prediction = model.predict(
    image_array,
    verbose=0
)

probability = float(prediction[0][0])


# ============================================================
# INTERPRET RESULT
# ============================================================

# Model classes:
#
# 0 = Disease
# 1 = Normal_No_Visible_Condition
#
# sigmoid output:
# close to 0 -> Disease
# close to 1 -> Normal

if probability >= 0.5:

    result = "Normal / No Visible Condition"

    confidence = probability * 100

else:

    result = "Disease"

    confidence = (1 - probability) * 100


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"\nPrediction       : {result}")
print(f"Confidence       : {confidence:.2f}%")
print(f"Raw model output : {probability:.4f}")


# ============================================================
# NEXT ACTION
# ============================================================

print("\n" + "-" * 70)

if result == "Disease":

    print(
        "Possible visible condition detected."
    )

    print(
        "Next step: send this image to the "
        "10-class disease model."
    )

else:

    print(
        "No visible condition detected."
    )

    print(
        "The disease classifier should NOT be called."
    )

print("-" * 70)


# ============================================================
# NOTE
# ============================================================

print("\nNOTE:")
print(
    "This is a project-level image classifier "
    "and is not a medical diagnosis."
)

print("\n" + "=" * 70)