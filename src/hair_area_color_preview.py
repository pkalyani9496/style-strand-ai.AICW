# ============================================================
# HAIR AREA COLOUR PREVIEW
# ============================================================
# Upload hair image
#        ↓
# Hair Segmentation Model
#        ↓
# Select a particular hair area
#        ↓
# Select hair colour
#        ↓
# Apply colour ONLY to selected hair area
#        ↓
# Save preview
# ============================================================


import os

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.widgets import RectangleSelector


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    "models"
    "\\hair_segmentation"
    "\\hair_segmentation_unet_best.keras"
)

OUTPUT_DIR = (
    "results"
    "\\hair_color_area"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# COLOUR OPTIONS
# ============================================================

COLOR_OPTIONS = {

    1: ("Burgundy", (128, 0, 32)),

    2: ("Purple", (128, 0, 128)),

    3: ("Copper", (184, 115, 51)),

    4: ("Caramel Brown", (150, 90, 45)),

    5: ("Chocolate Brown", (90, 45, 20)),

    6: ("Ash Brown", (105, 90, 80)),

    7: ("Honey Blonde", (210, 170, 80)),

    8: ("Black", (20, 20, 20)),

    9: ("Red", (180, 30, 30)),

    10: ("Blue", (30, 80, 200))
}


# ============================================================
# MODEL INPUT SIZE
# ============================================================

IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# COLOUR BLEND STRENGTH
# ============================================================

BLEND_STRENGTH = 0.65


# ============================================================
# GLOBAL VARIABLE
# ============================================================

selected_rectangle = None


# ============================================================
# LOAD SEGMENTATION MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING HAIR SEGMENTATION MODEL")
print("=" * 60)

print()

print(
    "Model path:"
)

print(
    MODEL_PATH
)

print()


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print(
    "Hair segmentation model loaded successfully."
)

print()


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    image_path
):

    print(
        "Loading image..."
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image:\n{image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ============================================================
# PREDICT HAIR MASK
# ============================================================

def predict_hair_mask(
    image
):

    print(
        "Predicting hair segmentation..."
    )

    original_height = image.shape[0]

    original_width = image.shape[1]


    # --------------------------------------------------------
    # Resize image for model
    # --------------------------------------------------------

    resized = cv2.resize(
        image,
        IMAGE_SIZE
    )


    # --------------------------------------------------------
    # Convert to float
    # --------------------------------------------------------

    resized = (
        resized.astype(
            np.float32
        )
        / 255.0
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    input_image = np.expand_dims(
        resized,
        axis=0
    )


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_image,
        verbose=0
    )


    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    prediction = prediction[0]


    # --------------------------------------------------------
    # Convert probability to binary mask
    # --------------------------------------------------------

    mask = (
        prediction[:, :, 0]
        > 0.5
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # Resize mask back to original image size
    # --------------------------------------------------------

    mask = cv2.resize(
        mask,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )


    return mask.astype(
        bool
    )


# ============================================================
# SELECT PARTICULAR HAIR AREA
# ============================================================

def select_area(
    image,
    hair_mask
):

    global selected_rectangle

    selected_rectangle = None


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )


    # --------------------------------------------------------
    # Display original image
    # --------------------------------------------------------

    ax.imshow(
        image.astype(
            np.uint8
        )
    )


    # --------------------------------------------------------
    # Display hair mask lightly
    # --------------------------------------------------------

    masked_hair = np.ma.masked_where(
        ~hair_mask,
        hair_mask
    )


    ax.imshow(
        masked_hair,
        alpha=0.20
    )


    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    ax.set_title(
        "Drag ONCE over the hair area you want to colour\n"
        "Then CLOSE this window"
    )


    ax.axis(
        "off"
    )


    # ========================================================
    # SELECTION CALLBACK
    # ========================================================

    def on_select(
        eclick,
        erelease
    ):

        global selected_rectangle


        # ----------------------------------------------------
        # Ignore additional selections
        # ----------------------------------------------------

        if selected_rectangle is not None:

            return


        # ----------------------------------------------------
        # Check coordinates
        # ----------------------------------------------------

        if (
            eclick.xdata is None
            or
            eclick.ydata is None
            or
            erelease.xdata is None
            or
            erelease.ydata is None
        ):

            return


        # ----------------------------------------------------
        # Calculate rectangle coordinates
        # ----------------------------------------------------

        x1 = int(
            min(
                eclick.xdata,
                erelease.xdata
            )
        )


        x2 = int(
            max(
                eclick.xdata,
                erelease.xdata
            )
        )


        y1 = int(
            min(
                eclick.ydata,
                erelease.ydata
            )
        )


        y2 = int(
            max(
                eclick.ydata,
                erelease.ydata
            )
        )


        # ----------------------------------------------------
        # Ignore very small accidental selections
        # ----------------------------------------------------

        if (
            x2 - x1 < 5
            or
            y2 - y1 < 5
        ):

            print()

            print(
                "Selection too small."
            )

            print(
                "Please drag a larger rectangle."
            )

            return


        # ----------------------------------------------------
        # Save selection
        # ----------------------------------------------------

        selected_rectangle = (
            x1,
            y1,
            x2,
            y2
        )


        print()

        print(
            "Selected area:"
        )

        print(
            f"X1={x1}, "
            f"Y1={y1}, "
            f"X2={x2}, "
            f"Y2={y2}"
        )


        # ----------------------------------------------------
        # Draw selected rectangle
        # ----------------------------------------------------

        from matplotlib.patches import Rectangle


        rectangle = Rectangle(
            (
                x1,
                y1
            ),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2
        )


        ax.add_patch(
            rectangle
        )


        ax.set_title(
            "Area selected!\n"
            "Close this window to continue."
        )


        fig.canvas.draw_idle()


    # ========================================================
    # RECTANGLE SELECTOR
    # ========================================================

    rectangle_selector = RectangleSelector(
        ax,
        on_select,

        useblit=True,

        button=[
            1
        ],

        minspanx=5,

        minspany=5,

        spancoords="pixels",

        interactive=False
    )


    # --------------------------------------------------------
    # Show selection window
    # --------------------------------------------------------

    plt.show()


    return selected_rectangle


# ============================================================
# CREATE SELECTED HAIR MASK
# ============================================================

def create_selected_hair_mask(
    hair_mask,
    rectangle
):

    if rectangle is None:

        return None


    x1, y1, x2, y2 = rectangle


    # --------------------------------------------------------
    # Create empty mask
    # --------------------------------------------------------

    selected_mask = np.zeros(
        hair_mask.shape,
        dtype=bool
    )


    # --------------------------------------------------------
    # Make sure coordinates are inside image
    # --------------------------------------------------------

    height, width = hair_mask.shape


    x1 = max(
        0,
        min(
            x1,
            width
        )
    )


    x2 = max(
        0,
        min(
            x2,
            width
        )
    )


    y1 = max(
        0,
        min(
            y1,
            height
        )
    )


    y2 = max(
        0,
        min(
            y2,
            height
        )
    )


    # --------------------------------------------------------
    # Rectangle ∩ Hair Mask
    # --------------------------------------------------------

    selected_mask[
        y1:y2,
        x1:x2
    ] = hair_mask[
        y1:y2,
        x1:x2
    ]


    return selected_mask


# ============================================================
# DISPLAY COLOUR OPTIONS
# ============================================================

def choose_colour():

    print()

    print(
        "=" * 60
    )

    print(
        "AVAILABLE HAIR COLOURS"
    )

    print(
        "=" * 60
    )

    print()


    for number, value in COLOR_OPTIONS.items():

        colour_name = value[0]

        print(
            f"{number}. {colour_name}"
        )


    print()


    while True:

        choice = input(
            "Enter colour number (1-10): "
        )


        try:

            choice = int(
                choice
            )

        except ValueError:

            print()

            print(
                "Please enter a number."
            )

            continue


        if choice in COLOR_OPTIONS:

            return COLOR_OPTIONS[
                choice
            ]


        print()

        print(
            "Invalid choice."
        )

        print(
            "Please select a number from 1 to 10."
        )


# ============================================================
# APPLY COLOUR
# ============================================================

def apply_colour(
    image,
    selected_mask,
    rgb_colour
):

    # --------------------------------------------------------
    # Create colour image
    # --------------------------------------------------------

    colour_layer = np.zeros_like(
        image,
        dtype=np.uint8
    )


    colour_layer[:, :] = rgb_colour


    # --------------------------------------------------------
    # Convert image to float
    # --------------------------------------------------------

    result = image.astype(
        np.float32
    )


    colour_layer = colour_layer.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Blend original hair with selected colour
    # --------------------------------------------------------

    blended = (
        result * (
            1.0 - BLEND_STRENGTH
        )
        +
        colour_layer * BLEND_STRENGTH
    )


    # --------------------------------------------------------
    # Apply ONLY selected hair pixels
    # --------------------------------------------------------

    result[
        selected_mask
    ] = blended[
        selected_mask
    ]


    # --------------------------------------------------------
    # Convert back to uint8
    # --------------------------------------------------------

    result = np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


    return result


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    image,
    path
):

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )


    cv2.imwrite(
        path,
        image_bgr
    )


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(
    mask,
    path
):

    mask_image = (
        mask.astype(
            np.uint8
        )
        * 255
    )


    cv2.imwrite(
        path,
        mask_image
    )


# ============================================================
# SHOW COMPARISON
# ============================================================

def save_comparison(
    original,
    preview,
    path
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7)
    )


    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[0].imshow(
        original
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis(
        "off"
    )


    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    axes[1].imshow(
        preview
    )

    axes[1].set_title(
        "Selected Hair Area Colour Preview"
    )

    axes[1].axis(
        "off"
    )


    plt.tight_layout()


    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()

    print(
        "=" * 60
    )

    print(
        "AI HAIR AREA COLOUR PREVIEW"
    )

    print(
        "=" * 60
    )

    print()


    # ========================================================
    # ASK IMAGE PATH
    # ========================================================

    image_path = input(
        "Enter full path of hair image:\n"
    ).strip()


    # Remove accidental quotes
    image_path = image_path.strip(
        '"'
    )

    image_path = image_path.strip(
        "'"
    )


    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if not os.path.exists(
        image_path
    ):

        print()

        print(
            "ERROR: Image file not found."
        )

        print()

        print(
            image_path
        )

        return


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = load_image(
        image_path
    )


    print()

    print(
        "Original image size:"
    )

    print(
        f"Width  : {image.shape[1]}"
    )

    print(
        f"Height : {image.shape[0]}"
    )


    # ========================================================
    # PREDICT HAIR MASK
    # ========================================================

    hair_mask = predict_hair_mask(
        image
    )


    # ========================================================
    # CALCULATE TOTAL HAIR AREA
    # ========================================================

    total_hair_pixels = np.sum(
        hair_mask
    )


    total_pixels = (
        hair_mask.shape[0]
        *
        hair_mask.shape[1]
    )


    hair_percentage = (
        total_hair_pixels
        /
        total_pixels
        *
        100
    )


    print()

    print(
        f"Detected hair area: "
        f"{hair_percentage:.2f}%"
    )


    # ========================================================
    # SELECT AREA
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "SELECT HAIR AREA"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "Instructions:"
    )

    print(
        "1. Drag ONE rectangle over the hair area."
    )

    print(
        "2. Make sure the rectangle covers some hair."
    )

    print(
        "3. After selection, CLOSE the image window."
    )

    print()


    rectangle = select_area(
        image,
        hair_mask
    )


    # ========================================================
    # CHECK SELECTION
    # ========================================================

    if rectangle is None:

        print()

        print(
            "No valid area was selected."
        )

        print(
            "Please run the program again."
        )

        return


    # ========================================================
    # CREATE SELECTED HAIR MASK
    # ========================================================

    selected_mask = create_selected_hair_mask(
        hair_mask,
        rectangle
    )


    if selected_mask is None:

        print()

        print(
            "Could not create selected hair mask."
        )

        return


    # ========================================================
    # SELECTED AREA PERCENTAGE
    # ========================================================

    selected_hair_pixels = np.sum(
        selected_mask
    )


    selected_hair_percentage = (
        selected_hair_pixels
        /
        total_pixels
        *
        100
    )


    print()

    print(
        "=" * 60
    )

    print(
        f"Selected hair area: "
        f"{selected_hair_percentage:.2f}%"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # CHECK ZERO AREA
    # ========================================================

    if selected_hair_pixels == 0:

        print()

        print(
            "WARNING:"
        )

        print(
            "The selected rectangle does not contain detected hair."
        )

        print()

        print(
            "Please run the program again and select"
        )

        print(
            "a rectangle directly over the hair."
        )

        return


    # ========================================================
    # CHOOSE COLOUR
    # ========================================================

    colour_name, rgb_colour = choose_colour()


    print()

    print(
        f"Selected colour: {colour_name}"
    )


    print(
        f"RGB value: {rgb_colour}"
    )


    # ========================================================
    # APPLY COLOUR
    # ========================================================

    preview = apply_colour(
        image,
        selected_mask,
        rgb_colour
    )


    # ========================================================
    # CREATE SAFE FILE NAME
    # ========================================================

    safe_colour_name = (
        colour_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    output_image_path = os.path.join(
        OUTPUT_DIR,
        f"selected_area_{safe_colour_name}.jpg"
    )


    output_mask_path = os.path.join(
        OUTPUT_DIR,
        "selected_hair_area_mask.png"
    )


    comparison_path = os.path.join(
        OUTPUT_DIR,
        f"selected_area_preview_{safe_colour_name}.png"
    )


    # ========================================================
    # SAVE PREVIEW
    # ========================================================

    save_image(
        preview,
        output_image_path
    )


    # ========================================================
    # SAVE MASK
    # ========================================================

    save_mask(
        selected_mask,
        output_mask_path
    )


    # ========================================================
    # SAVE COMPARISON
    # ========================================================

    save_comparison(
        image,
        preview,
        comparison_path
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "COLOUR PREVIEW COMPLETE"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "Selected colour:"
    )

    print(
        colour_name
    )

    print()

    print(
        "Selected hair area:"
    )

    print(
        f"{selected_hair_percentage:.2f}%"
    )

    print()

    print(
        "Final preview saved:"
    )

    print(
        output_image_path
    )

    print()

    print(
        "Selected mask saved:"
    )

    print(
        output_mask_path
    )

    print()

    print(
        "Comparison saved:"
    )

    print(
        comparison_path
    )

    print()

    print(
        "=" * 60
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()