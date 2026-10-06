import base64
import io
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_SIZE = (224, 224)

MODEL_PATHS = {
    "presence": ROOT_DIR / "models/hair_presence_v2/hair_presence_mobilenetv2_v2.keras",
    "presence_classes": ROOT_DIR / "models/hair_presence_v2/class_names.json",
    "type": ROOT_DIR / "models/hair_type/hair_type_mobilenetv2.keras",
    "type_classes": ROOT_DIR / "models/hair_type/class_names.json",
    "segmentation": ROOT_DIR / "models/hair_segmentation/hair_segmentation_unet_best.keras",
    "condition_gate": ROOT_DIR / "models/hair_condition_gate/best_hair_condition_gate_v2.keras",
    "condition_classes": ROOT_DIR / "models/hair_condition_gate/class_names.json",
    "disease": ROOT_DIR / "models/hair_disease/best_hair_disease_model.keras",
    "disease_classes": ROOT_DIR / "models/hair_disease/class_names.json",
}

COLOR_OPTIONS = {
    "Espresso": [55, 35, 28],
    "Soft black": [38, 34, 33],
    "Blue black": [31, 43, 58],
    "Dark chocolate": [78, 47, 35],
    "Mocha": [112, 76, 60],
    "Chestnut": [132, 76, 48],
    "Mahogany": [112, 46, 45],
    "Burgundy": [128, 38, 58],
    "Cherry cola": [105, 35, 46],
    "Copper": [177, 91, 53],
    "Auburn": [151, 68, 43],
    "Cinnamon": [171, 104, 70],
    "Caramel": [183, 125, 72],
    "Toffee": [173, 119, 80],
    "Honey blonde": [205, 164, 95],
    "Golden blonde": [214, 177, 95],
    "Butter blonde": [224, 198, 139],
    "Champagne blonde": [218, 202, 168],
    "Ash blonde": [166, 157, 142],
    "Mushroom brown": [130, 119, 105],
    "Rose brown": [151, 102, 96],
    "Dusty rose": [190, 119, 132],
    "Pastel pink": [224, 157, 178],
    "Vivid magenta": [210, 36, 121],
    "Violet": [115, 77, 145],
    "Lavender": [159, 137, 184],
    "Midnight blue": [43, 66, 113],
    "Denim blue": [74, 115, 157],
    "Teal": [48, 126, 119],
    "Emerald": [47, 111, 80],
    "Silver": [176, 180, 184],
    "Platinum": [215, 212, 202],
    "Fire red": [176, 50, 47],
}

HAIRCUTS = {
    "Curly Hair": [
        {
            "name": "Curly shag",
            "length": "medium",
            "presentation": "Feminine",
            "description": "Airy, rounded layers let curls keep their shape while adding movement.",
            "detail": "Great for natural volume",
            "image": "short-curly-hair-layered-bob-black-color.jpg",
        },
        {
            "name": "Long curly layers",
            "length": "long",
            "presentation": "Feminine",
            "description": "Long, curl-aware layers reduce bulk without losing length.",
            "detail": "Keeps length and bounce",
            "image": "long-layered-curly-hair-6.jpg",
        },
        {
            "name": "Curly butterfly layers",
            "length": "long",
            "presentation": "Feminine",
            "description": "Long, rounded layers lift the crown and let the curls fan out below the cheekbones.",
            "detail": "Crown lift with length kept",
            "image": "long-curly-hairstyle-for-indian-women-3.jpg",
        },
        {
            "name": "Defined curl curtain",
            "length": "long",
            "presentation": "Feminine",
            "description": "A soft center part and cheekbone-length front layers frame the face without cutting the back short.",
            "detail": "Face-framing, not a full fringe",
            "image": "99dd012b20e25bb446a8afdca933dd74--long-curly-hair-naturally-curly-hair.jpg",
        },
        {
            "name": "Soft curly fringe",
            "length": "medium",
            "presentation": "Feminine",
            "description": "A diffused fringe blends into the curl pattern for a softer frame.",
            "detail": "A little styling, lots of shape",
            "image": "0dba94f77ba91fe4c546048f7476c0ed--curly-hair-cuts-bangs-curly-hair-fringe.jpg",
        },
        {
            "name": "Rounded curly bob",
            "length": "short",
            "presentation": "Feminine",
            "description": "A cheek-to-jaw-length rounded outline gives curls shape while keeping the sides softly full.",
            "detail": "A defined shorter silhouette",
            "image": "c26f92d7ae9e9f485ccdc735e2af2b1c--short-curly-hair-black-bangs-curly-hair.jpg",
        },
        {
            "name": "Curly textured crop",
            "length": "short",
            "presentation": "Masculine",
            "description": "A compact shape keeps natural curls defined, with soft length left on top.",
            "detail": "Short sides, natural curl on top",
            "image": "curly-hair-man.jpg",
        },
        {
            "name": "Medium-length curl flow",
            "length": "medium",
            "presentation": "Masculine",
            "description": "Loose, grown-out layers let curls move while keeping the sides relaxed.",
            "detail": "A longer, low-fuss shape",
            "image": "besthairstylesforindianmen15_1378796017.jpg",
        },
    ],
    "Straight Hair": [
        {
            "name": "Butterfly layers",
            "length": "long",
            "presentation": "Feminine",
            "description": "Long, face-framing layers create lift and movement without sacrificing length.",
            "detail": "Movement with long length",
            "image": "Highlighted-Straight-Hair-with-Long-Layers.jpg",
        },
        {
            "name": "Blunt collarbone lob",
            "length": "medium",
            "presentation": "Feminine",
            "description": "A clean, one-length shape makes straight hair look polished and full.",
            "detail": "Sharp, easy-to-style outline",
            "image": "Haircut-Names-With-Pictures-For-Females-Or-Girls.jpg",
        },
        {
            "name": "Face-framing layers",
            "length": "long",
            "presentation": "Feminine",
            "description": "Subtle front layers soften the silhouette and keep the ends looking full.",
            "detail": "A low-commitment refresh",
            "image": "8-centreparted-layered-cut-for-long-hair.jpg",
        },
        {
            "name": "Long U-shaped layers",
            "length": "long",
            "presentation": "Feminine",
            "description": "A softly rounded perimeter keeps the length while long internal layers add swing to straight hair.",
            "detail": "Full ends with lighter movement",
            "image": "layered-haircut-with-balayage-long-straight.jpg",
        },
        {
            "name": "Sleek one-length finish",
            "length": "long",
            "presentation": "Feminine",
            "description": "A clean, mostly one-length outline emphasizes shine and makes the ends look dense.",
            "detail": "Minimal layers, polished shape",
            "image": "long-straight-hairstyles.jpg",
        },
        {
            "name": "Textured pixie",
            "length": "short",
            "presentation": "Feminine",
            "description": "A cropped pixie with a longer, piecey top gives straight hair a lighter, more directional shape.",
            "detail": "Short with styling flexibility",
            "image": "Edgy-Pixie-Haircuts-Straight-Short-Hair.jpg",
        },
        {
            "name": "Textured crew cut",
            "length": "short",
            "presentation": "Masculine",
            "description": "A short, clean outline with a little length left on top for easy texture.",
            "detail": "Short, tidy, and low maintenance",
            "image": "crew-cut-for-men.jpg",
        },
        {
            "name": "Short textured crop",
            "length": "short",
            "presentation": "Masculine",
            "description": "A close fade paired with a softly textured top keeps straight hair light.",
            "detail": "Neat sides with a flexible top",
            "image": "clean-high-fade-for-men-with-thick-hair-500x625.jpg",
        },
        {
            "name": "Short spiky crop",
            "length": "short",
            "presentation": "Masculine",
            "description": "A cropped shape adds definition while keeping the routine simple.",
            "detail": "A sharper, shorter profile",
            "image": "1-short-spiky-mens-haircut.jpg",
        },
    ],
    "Wavy Hair": [
        {
            "name": "Textured wolf cut",
            "length": "medium",
            "presentation": "Feminine",
            "description": "Shaggy, blended layers bring out natural waves and lift at the crown.",
            "detail": "Defined texture and volume",
            "image": "Layered-wavy-hairstyles4-1.png",
        },
        {
            "name": "Wavy collarbone lob",
            "length": "medium",
            "presentation": "Feminine",
            "description": "A relaxed mid-length shape gives waves room to form without feeling heavy.",
            "detail": "Air-dry friendly length",
            "image": "Bouncy-short-hairstyle.jpg",
        },
        {
            "name": "Soft long layers",
            "length": "long",
            "presentation": "Feminine",
            "description": "Light layers encourage loose movement while keeping the overall length.",
            "detail": "Natural movement, less weight",
            "image": "karishma-tanna-hairstyles-iDiva-Thumbnail_5e3c0b17b1948.jpg",
        },
        {
            "name": "Long butterfly waves",
            "length": "long",
            "presentation": "Feminine",
            "description": "Long, sweeping face-framing layers lift the front while keeping the back comfortably long.",
            "detail": "Soft lift around the face",
            "image": "beauty-2012-05-65-100-best-hairstyles-aishwarya-rai-long-layers-main.jpg",
        },
        {
            "name": "Long lived-in waves",
            "length": "long",
            "presentation": "Feminine",
            "description": "Long blended layers keep the outline relaxed and let natural waves start below the shoulders.",
            "detail": "Low-fuss, airy movement",
            "image": "Messy-Long-Hair.jpg",
        },
        {
            "name": "Rounded wavy bob",
            "length": "short",
            "presentation": "Feminine",
            "description": "A jaw-length rounded bob gives waves a clear shape, with softly textured ends instead of a blunt block.",
            "detail": "A shorter shape with movement",
            "image": "Bouncy-short-hairstyle(1).jpg",
        },
        {
            "name": "Wavy side-swept crop",
            "length": "short",
            "presentation": "Masculine",
            "description": "A relaxed side sweep works with natural bends instead of fighting them.",
            "detail": "Soft movement with a clean outline",
            "image": "Wavy-Short-Hairstyles-for-Men.jpg",
        },
        {
            "name": "Wavy low taper",
            "length": "short",
            "presentation": "Masculine",
            "description": "Tapered sides keep the silhouette neat while the wavy top stays loose.",
            "detail": "Short sides, natural texture",
            "image": "comb-over-haircuts-men-wavy-fade-334x500.jpg",
        },
        {
            "name": "Long side-swept waves",
            "length": "long",
            "presentation": "Masculine",
            "description": "A longer side sweep leaves room for natural wave and a softer finish.",
            "detail": "A longer, relaxed shape",
            "image": "Wavy-Side-Swept.jpg",
        },
    ],
}


def _load_keras_model(model_path):
    def compatible_from_config(ignored_fields, none_only=(), false_only=()):
        @classmethod
        def from_config(cls, config):
            config = dict(config)
            for field in none_only:
                if config.get(field) is not None:
                    raise ValueError(
                        f"Cannot load {model_path}: unsupported non-default "
                        f"{field}={config[field]!r}"
                    )
            for field in false_only:
                if config.get(field, False) is not False:
                    raise ValueError(
                        f"Cannot load {model_path}: unsupported enabled "
                        f"{field} configuration"
                    )
            for field in ignored_fields:
                config.pop(field, None)
            return cls(**config)

        return from_config

    # These default-only fields were added after the Keras version available
    # for the app runtime; they do not affect inference for these saved models.
    with ExitStack() as patches:
        patches.enter_context(
            patch.object(
                tf.keras.initializers.GlorotUniform,
                "from_config",
                compatible_from_config(
                    ("input_axes", "output_axes"),
                    none_only=("input_axes", "output_axes"),
                ),
            )
        )
        patches.enter_context(
            patch.object(
                tf.keras.layers.BatchNormalization,
                "from_config",
                compatible_from_config(
                    ("renorm", "renorm_clipping", "renorm_momentum"),
                    false_only=("renorm",),
                ),
            )
        )
        patches.enter_context(
            patch.object(
                tf.keras.layers.Dense,
                "from_config",
                compatible_from_config(
                    ("quantization_config",),
                    none_only=("quantization_config",),
                ),
            )
        )
        return tf.keras.models.load_model(model_path, compile=False)


@st.cache_resource
def load_models():
    with MODEL_PATHS["presence_classes"].open("r", encoding="utf-8") as file:
        presence_classes = json.load(file)
    with MODEL_PATHS["type_classes"].open("r", encoding="utf-8") as file:
        type_classes = json.load(file)
    with MODEL_PATHS["condition_classes"].open("r", encoding="utf-8") as file:
        condition_classes = json.load(file)
    with MODEL_PATHS["disease_classes"].open("r", encoding="utf-8") as file:
        disease_classes = json.load(file)

    return {
        "presence": _load_keras_model(MODEL_PATHS["presence"]),
        "presence_classes": presence_classes,
        "type": _load_keras_model(MODEL_PATHS["type"]),
        "type_classes": type_classes,
        "condition_gate": _load_keras_model(MODEL_PATHS["condition_gate"]),
        "condition_classes": condition_classes,
        "disease": _load_keras_model(MODEL_PATHS["disease"]),
        "disease_classes": disease_classes,
        "segmentation": _load_keras_model(MODEL_PATHS["segmentation"]),
    }


def _image_from_data_url(image_data):
    encoded_image = image_data.split(",", 1)[-1]
    image_bytes = base64.b64decode(encoded_image)
    return ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")


def _png_data_url(image_array):
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG", optimize=True)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


def _clean_hair_mask(predicted_mask, output_size):
    width, height = output_size
    probability = cv2.resize(
        predicted_mask.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    probability = np.clip(probability, 0, 1)
    binary_mask = (probability >= 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    )
    min_component_area = max(16, int(width * height * 0.0002))
    clean_mask = np.zeros_like(binary_mask)
    for component_index in range(1, component_count):
        component_area = component_stats[component_index, cv2.CC_STAT_AREA]
        if component_area >= min_component_area:
            clean_mask[component_labels == component_index] = 1

    confidence = np.clip((probability - 0.15) / 0.7, 0, 1)
    alpha = cv2.GaussianBlur(clean_mask * confidence, (0, 0), 0.65)
    return clean_mask, np.clip(alpha * 255, 0, 255).astype(np.uint8)


def _visible_hair_length(mask):
    hair_rows = np.flatnonzero(np.any(mask > 0, axis=1))
    if hair_rows.size == 0:
        return "unclear"

    visible_height = (hair_rows[-1] - hair_rows[0] + 1) / mask.shape[0]
    if visible_height < 0.36:
        return "short"
    if visible_height < 0.62:
        return "medium"
    return "long"


def analyze_image(image_data, models):
    image = _image_from_data_url(image_data)
    original_image = np.asarray(image)
    classifier_image = np.asarray(image.resize(IMAGE_SIZE), dtype=np.float32)
    classifier_input = np.expand_dims(classifier_image, axis=0)

    presence_scores = models["presence"].predict(classifier_input, verbose=0)[0]
    presence_classes = models["presence_classes"]
    hair_index = presence_classes.index("Hair")
    nonhair_index = presence_classes.index("NonHair")
    presence_index = int(np.argmax(presence_scores))
    hair_detected = presence_classes[presence_index] == "Hair"

    result = {
        "imageData": image_data,
        "hairDetected": hair_detected,
        "hairConfidence": float(presence_scores[hair_index]),
        "nonHairConfidence": float(presence_scores[nonhair_index]),
        "presenceConfidence": float(presence_scores[presence_index]),
    }

    normal_index = models["condition_classes"].index(
        "Normal_No_Visible_Condition"
    )
    normal_probability = float(
        models["condition_gate"].predict(classifier_input, verbose=0)[0][0]
    )
    condition_scores = [0.0] * len(models["condition_classes"])
    condition_scores[normal_index] = normal_probability
    condition_scores[1 - normal_index] = 1.0 - normal_probability
    condition_index = int(np.argmax(condition_scores))
    condition_label = models["condition_classes"][condition_index]

    disease_scores = []
    disease_label = ""
    disease_confidence = 0.0
    if condition_label == "Disease":
        disease_scores = [
            float(score)
            for score in models["disease"].predict(
                classifier_input,
                verbose=0,
            )[0]
        ]
        disease_index = int(np.argmax(disease_scores))
        disease_label = models["disease_classes"][disease_index]
        disease_confidence = disease_scores[disease_index]

    condition_result = {
        "conditionEvaluated": True,
        "conditionLabel": condition_label,
        "conditionConfidence": float(condition_scores[condition_index]),
        "conditionClasses": models["condition_classes"],
        "conditionScores": condition_scores,
        "diseaseLabel": disease_label,
        "diseaseConfidence": disease_confidence,
        "diseaseClasses": models["disease_classes"],
        "diseaseScores": disease_scores,
    }
    if not hair_detected:
        return {**result, **condition_result}

    type_scores = models["type"].predict(classifier_input, verbose=0)[0]
    type_classes = models["type_classes"]
    type_index = int(np.argmax(type_scores))

    segmentation_image = cv2.resize(original_image, IMAGE_SIZE)
    segmentation_input = np.expand_dims(
        segmentation_image.astype(np.float32) / 255.0,
        axis=0,
    )
    predicted_mask = models["segmentation"].predict(
        segmentation_input,
        verbose=0,
    )[0, :, :, 0]
    mask, mask_alpha = _clean_hair_mask(
        predicted_mask,
        (original_image.shape[1], original_image.shape[0]),
    )
    hair_area = float(np.mean(mask) * 100)

    return {
        **result,
        **condition_result,
        "hairType": type_classes[type_index],
        "typeConfidence": float(type_scores[type_index]),
        "typeScores": [float(score) for score in type_scores],
        "typeClasses": type_classes,
        "hairArea": hair_area,
        "visibleHairLength": _visible_hair_length(mask),
        "maskData": _png_data_url(mask_alpha),
        "imageWidth": int(original_image.shape[1]),
        "imageHeight": int(original_image.shape[0]),
    }


@st.cache_data(show_spinner=False)
def _load_preview_assets(image_path, _segmentation_model):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image.thumbnail((720, 620), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    preview_buffer = io.BytesIO()
    image.save(preview_buffer, format="JPEG", quality=82, optimize=True)
    preview_data = base64.b64encode(preview_buffer.getvalue()).decode("ascii")

    segmentation_input = cv2.resize(image_array, IMAGE_SIZE).astype(np.float32) / 255.0
    predicted_mask = _segmentation_model.predict(
        np.expand_dims(segmentation_input, axis=0),
        verbose=0,
    )[0, :, :, 0]
    _, alpha = _clean_hair_mask(predicted_mask, (image.width, image.height))
    visible_pixels = np.argwhere(alpha > 96)
    if visible_pixels.size == 0:
        return f"data:image/jpeg;base64,{preview_data}", ""

    top, left = visible_pixels.min(axis=0)
    bottom, right = visible_pixels.max(axis=0) + 1
    padding = max(4, int(max(bottom - top, right - left) * 0.06))
    top = max(0, top - padding)
    left = max(0, left - padding)
    bottom = min(image.height, bottom + padding)
    right = min(image.width, right + padding)

    rgba = np.dstack((image_array, alpha))
    hair_image = Image.fromarray(rgba).crop((left, top, right, bottom))
    hair_buffer = io.BytesIO()
    hair_image.save(hair_buffer, format="PNG", optimize=True)
    hair_data = base64.b64encode(hair_buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{preview_data}", f"data:image/png;base64,{hair_data}"


def get_haircut_recommendations(hair_type, segmentation_model):
    recommendations = []
    class_directory = ROOT_DIR / "datasets/hair_type_prepared/train" / hair_type

    for haircut in HAIRCUTS.get(hair_type, []):
        image_path = class_directory / haircut["image"]
        recommendation = {
            key: value for key, value in haircut.items() if key != "image"
        }
        if image_path.is_file():
            recommendation["preview"], recommendation["hairData"] = (
                _load_preview_assets(str(image_path), segmentation_model)
            )
        else:
            recommendation["preview"] = ""
            recommendation["hairData"] = ""
        recommendations.append(recommendation)

    return recommendations