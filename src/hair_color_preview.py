import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

MODEL_PATH = (
    r"models\hair_segmentation"
    r"\hair_segmentation_unet_best.keras"
)

RESULT_DIR = r"results\hair_color"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# COLOUR OPTIONS
# ============================================================

HAIR_COLORS = {

    "Burgundy": (128, 0, 32),

    "Purple": (128, 0, 128),

    "Copper": (184, 115, 51),

    "Caramel Brown": (150, 95, 45),

    "Chocolate Brown": (90, 45, 20),

    "Ash Brown": (100, 85, 75),

    "Honey Blonde": (210, 170, 80),

    "Black": (20, 20, 20),

    "Red": (180, 30, 30),

    "Blue": (40, 80, 180)
}


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    original = Image.open(
        image_path
    ).convert("RGB")

    resized = original.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.array(
        resized,
        dtype=np.float32
    )

    # Model expects 0-1
    model_input = (
        image_array / 255.0
    )

    return (
        original,
        image_array,
        model_input
    )


# ============================================================
# CREATE HAIR MASK
# ============================================================

def create_hair_mask(
    model,
    model_input
):

    input_batch = np.expand_dims(
        model_input,
        axis=0
    )

    prediction = model.predict(
        input_batch,
        verbose=0
    )[0]

    # Convert probability to binary mask
    mask = (
        prediction[:, :, 0] >= 0.5
    )

    return mask


# ============================================================
# APPLY HAIR COLOUR
# ============================================================

def apply_hair_color(
    image,
    hair_mask,
    target_color,
    strength=0.65
):

    result = image.copy()

    color = np.array(
        target_color,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Colour blending
    # --------------------------------------------------------

    mask_indices = hair_mask

    original_pixels = (
        result[mask_indices]
    )

    coloured_pixels = (
        original_pixels * (1 - strength)
        +
        color * strength
    )

    result[mask_indices] = (
        coloured_pixels
    )

    return np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# SAVE PREVIEW
# ============================================================

def save_preview(
    original_image,
    hair_mask,
    coloured_image,
    colour_name
):

    # --------------------------------------------------------
    # Save coloured image
    # --------------------------------------------------------

    output_name = (
        "hair_color_"
        + colour_name.lower()
        .replace(" ", "_")
        + ".jpg"
    )

    output_path = os.path.join(
        RESULT_DIR,
        output_name
    )

    Image.fromarray(
        coloured_image
    ).save(
        output_path
    )

    # --------------------------------------------------------
    # Create mask visualization
    # --------------------------------------------------------

    mask_image = (
        hair_mask.astype(np.uint8)
        * 255
    )

    mask_image = Image.fromarray(
        mask_image
    )

    mask_path = os.path.join(
        RESULT_DIR,
        "hair_mask.png"
    )

    mask_image.save(
        mask_path
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 4)
    )

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original_image
    )

    plt.title(
        "Original Image"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        hair_mask,
        cmap="gray"
    )

    plt.title(
        "Detected Hair"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        coloured_image
    )

    plt.title(
        colour_name
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    figure_path = os.path.join(
        RESULT_DIR,
        "hair_color_preview_"
        + colour_name.lower()
        .replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Colour preview saved:"
    )

    print(
        output_path
    )

    print()
    print(
        "Preview comparison saved:"
    )

    print(
        figure_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VIRTUAL HAIR COLOUR PREVIEW")
    print("=" * 70)

    # --------------------------------------------------------
    # Ask for image
    # --------------------------------------------------------

    image_path = input(
        "\nEnter the full path of your hair image:\n"
    ).strip()

    if not os.path.exists(
        image_path
    ):

        print()
        print(
            "ERROR: Image not found."
        )

        return

    # --------------------------------------------------------
    # Load segmentation model
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print()
        print(
            "ERROR: Hair segmentation model not found."
        )

        print(
            MODEL_PATH
        )

        return

    print()
    print(
        "Loading hair segmentation model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    print()
    print(
        "Processing image..."
    )

    original_image, image_array, model_input = (
        load_image(image_path)
    )

    # --------------------------------------------------------
    # Generate hair mask
    # --------------------------------------------------------

    print()
    print(
        "Detecting hair area..."
    )

    hair_mask = create_hair_mask(
        model,
        model_input
    )

    hair_percentage = (
        np.mean(hair_mask)
        * 100
    )

    print()
    print(
        f"Detected hair area: "
        f"{hair_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Show colour options
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print("AVAILABLE HAIR COLOURS")
    print("=" * 50)

    colour_names = list(
        HAIR_COLORS.keys()
    )

    for i, name in enumerate(
        colour_names,
        start=1
    ):

        print(
            f"{i}. {name}"
        )

    # --------------------------------------------------------
    # User selects colour
    # --------------------------------------------------------

    while True:

        choice = input(
            "\nEnter colour number: "
        ).strip()

        try:

            choice_number = int(
                choice
            )

            if (
                1
                <= choice_number
                <= len(colour_names)
            ):
                break

            print(
                "Please enter a valid number."
            )

        except ValueError:

            print(
                "Please enter a number."
            )

    colour_name = colour_names[
        choice_number - 1
    ]

    target_color = HAIR_COLORS[
        colour_name
    ]

    print()
    print(
        "Selected colour:",
        colour_name
    )

    # --------------------------------------------------------
    # Apply colour
    # --------------------------------------------------------

    print()
    print(
        "Applying colour only to detected hair..."
    )

    coloured_image = apply_hair_color(
        image_array,
        hair_mask,
        target_color,
        strength=0.65
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    save_preview(
        original_image,
        hair_mask,
        coloured_image,
        colour_name
    )

    print()
    print("=" * 70)
    print("VIRTUAL HAIR COLOUR PREVIEW COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()