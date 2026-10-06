import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "hair_disease",
    "hair_disease_mobilenetv2.keras"
)

CLASS_NAMES_PATH = os.path.join(
    "models",
    "hair_disease",
    "class_names.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Hair Disease model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Hair Disease model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Classes:")
for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image_path):
    """
    Load and prepare an image for the MobileNetV2 model.
    """

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# HAIR DISEASE PREDICTION
# ============================================================

def predict_hair_condition(image_path):
    """
    Predict the hair/scalp condition from an image.

    Returns:
        condition_name
        confidence
        all_probabilities
    """

    image = prepare_image(image_path)

    predictions = model.predict(image, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))

    condition_name = class_names[predicted_index]

    confidence = float(predictions[predicted_index])

    probabilities = {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

    return condition_name, confidence, probabilities


# ============================================================
# GENERAL INFORMATION
# ============================================================

CONDITION_INFO = {

    "Alopecia Areata": {
        "description": "The model classified the image as Alopecia Areata.",
        "care": [
            "Consider discussing the finding with a qualified dermatologist.",
            "Avoid excessive pulling or tight hairstyles.",
            "Use gentle hair and scalp care practices."
        ]
    },

    "Contact Dermatitis": {
        "description": "The model classified the image as Contact Dermatitis.",
        "care": [
            "Avoid products that appear to irritate the scalp.",
            "Use gentle hair-care products.",
            "Consider professional medical advice if irritation persists."
        ]
    },

    "Folliculitis": {
        "description": "The model classified the image as Folliculitis.",
        "care": [
            "Keep the scalp clean and avoid unnecessary scratching.",
            "Avoid sharing combs, brushes, or other personal hair-care items.",
            "Consider consulting a healthcare professional for persistent symptoms."
        ]
    },

    "Head Lice": {
        "description": "The model classified the image as Head Lice.",
        "care": [
            "Avoid sharing combs, brushes, hats, or pillows.",
            "Check close household contacts if appropriate.",
            "Consult a healthcare professional or pharmacist about appropriate treatment."
        ]
    },

    "Lichen Planus": {
        "description": "The model classified the image as Lichen Planus.",
        "care": [
            "Avoid scratching or irritating the scalp.",
            "Use gentle hair-care products.",
            "Consider evaluation by a dermatologist."
        ]
    },

    "Male Pattern Baldness": {
        "description": "The model classified the image as Male Pattern Baldness.",
        "care": [
            "Avoid excessive tension from tight hairstyles.",
            "Maintain gentle scalp-care habits.",
            "Consider discussing persistent or progressive hair loss with a dermatologist."
        ]
    },

    "Psoriasis": {
        "description": "The model classified the image as Psoriasis.",
        "care": [
            "Avoid scratching or picking scalp scales.",
            "Use gentle scalp-care products.",
            "Consider professional medical advice for persistent symptoms."
        ]
    },

    "Seborrheic Dermatitis": {
        "description": "The model classified the image as Seborrheic Dermatitis.",
        "care": [
            "Keep the scalp clean with gentle hair-care practices.",
            "Avoid scratching irritated areas.",
            "Consider professional advice if symptoms are persistent or severe."
        ]
    },

    "Telogen Effluvium": {
        "description": "The model classified the image as Telogen Effluvium.",
        "care": [
            "Avoid excessive chemical or heat treatments.",
            "Use gentle hair-care practices.",
            "Consider discussing significant or persistent hair shedding with a healthcare professional."
        ]
    },

    "Tinea Capitis": {
        "description": "The model classified the image as Tinea Capitis.",
        "care": [
            "Avoid sharing combs, brushes, hats, or other personal items.",
            "Keep personal hair-care items clean.",
            "Seek professional medical advice because scalp fungal infections may require appropriate treatment."
        ]
    }
}


# ============================================================
# GET CONDITION INFORMATION
# ============================================================

def get_condition_info(condition_name):

    return CONDITION_INFO.get(
        condition_name,
        {
            "description": "The model detected a hair/scalp condition.",
            "care": [
                "Consider consulting a qualified healthcare professional."
            ]
        }
    )


# ============================================================
# TEST FUNCTION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("HAIR HEALTH / CONDITION PREDICTION")
    print("=" * 60)

    image_path = input("\nEnter image path: ").strip().strip('"')

    if not os.path.exists(image_path):
        print("\nERROR: Image file not found.")
        exit()

    condition, confidence, probabilities = predict_hair_condition(
        image_path
    )

    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)

    print(f"\nCondition : {condition}")
    print(f"Confidence: {confidence * 100:.2f}%")

    info = get_condition_info(condition)

    print("\nDescription:")
    print(info["description"])

    print("\nGeneral Care Information:")

    for tip in info["care"]:
        print(f"- {tip}")

    print("\nAll Class Probabilities:")

    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for name, probability in sorted_probabilities:
        print(f"{name:25s}: {probability * 100:.2f}%")

    print("\n" + "=" * 60)