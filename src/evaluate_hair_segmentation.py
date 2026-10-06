import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

TEST_IMAGE_DIR = r"datasets\hair_segmentation_prepared\test\images"
TEST_MASK_DIR = r"datasets\hair_segmentation_prepared\test\masks"

MODEL_PATH = r"models\hair_segmentation\hair_segmentation_unet_best.keras"

RESULT_DIR = r"results\hair_segmentation"

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):
    """
    Load image and resize to model input size.

    Pixel values:
    0-255  -->  0-1
    """

    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    return image


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(mask_path):
    """
    Load ground-truth hair mask.

    Background = 0
    Hair       = 1
    """

    mask = Image.open(mask_path).convert("L")

    mask = mask.resize(
        IMAGE_SIZE,
        Image.Resampling.NEAREST
    )

    mask = np.array(
        mask,
        dtype=np.float32
    )

    # Convert 0-255 to 0-1
    mask = mask / 255.0

    # Binary mask
    mask = (
        mask >= 0.5
    ).astype(np.float32)

    # Add channel dimension
    mask = np.expand_dims(
        mask,
        axis=-1
    )

    return mask


# ============================================================
# NORMALIZE FILE NAME
# ============================================================

def normalize_name(filename):
    """
    Convert image and mask filenames to the same base name.

    Examples:

    Frame00010-org.jpg
        ->
    frame00010

    Frame00010-gt.pbm
        ->
    frame00010
    """

    name = os.path.splitext(filename)[0]

    name = name.lower()

    # Remove original-image suffix
    if name.endswith("-org"):
        name = name[:-4]

    # Remove ground-truth mask suffix
    if name.endswith("-gt"):
        name = name[:-3]

    return name


# ============================================================
# FIND IMAGE-MASK PAIRS
# ============================================================

def get_image_mask_pairs():

    image_files = sorted([
        f
        for f in os.listdir(TEST_IMAGE_DIR)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    mask_files = sorted([
        f
        for f in os.listdir(TEST_MASK_DIR)
        if f.lower().endswith(
            (".pbm", ".png", ".jpg", ".jpeg")
        )
    ])

    print()
    print("=" * 60)
    print("TEST DATA")
    print("=" * 60)

    print(
        "Test images found :",
        len(image_files)
    )

    print(
        "Test masks found  :",
        len(mask_files)
    )

    # --------------------------------------------------------
    # Create mask lookup dictionary
    # --------------------------------------------------------

    mask_lookup = {}

    for mask_file in mask_files:

        normalized = normalize_name(
            mask_file
        )

        mask_lookup[normalized] = mask_file

    # --------------------------------------------------------
    # Match images with masks
    # --------------------------------------------------------

    pairs = []

    unmatched_images = []

    for image_file in image_files:

        normalized = normalize_name(
            image_file
        )

        if normalized in mask_lookup:

            mask_file = mask_lookup[
                normalized
            ]

            image_path = os.path.join(
                TEST_IMAGE_DIR,
                image_file
            )

            mask_path = os.path.join(
                TEST_MASK_DIR,
                mask_file
            )

            pairs.append(
                (
                    image_path,
                    mask_path
                )
            )

        else:

            unmatched_images.append(
                image_file
            )

    print(
        "Matched image-mask pairs :",
        len(pairs)
    )

    print(
        "Unmatched images         :",
        len(unmatched_images)
    )

    # --------------------------------------------------------
    # Show first few matches
    # --------------------------------------------------------

    if len(pairs) > 0:

        print()
        print("First 5 matched pairs:")

        for image_path, mask_path in pairs[:5]:

            print(
                "Image:",
                os.path.basename(image_path)
            )

            print(
                "Mask :",
                os.path.basename(mask_path)
            )

            print()

    # --------------------------------------------------------
    # Show unmatched files
    # --------------------------------------------------------

    if len(unmatched_images) > 0:

        print()
        print("First unmatched images:")

        for filename in unmatched_images[:10]:

            print(filename)

    return pairs


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(y_true, y_pred):

    y_true = y_true.astype(
        np.float32
    )

    y_pred = y_pred.astype(
        np.float32
    )

    intersection = np.sum(
        y_true * y_pred
    )

    dice = (
        2.0 * intersection + 1e-7
    ) / (
        np.sum(y_true)
        +
        np.sum(y_pred)
        +
        1e-7
    )

    return dice


# ============================================================
# IOU SCORE
# ============================================================

def iou_score(y_true, y_pred):

    y_true = y_true.astype(
        np.float32
    )

    y_pred = y_pred.astype(
        np.float32
    )

    intersection = np.sum(
        y_true * y_pred
    )

    union = (
        np.sum(y_true)
        +
        np.sum(y_pred)
        -
        intersection
    )

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    return iou


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HAIR SEGMENTATION MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print()
        print("ERROR: Model not found!")

        print(
            "Expected model:"
        )

        print(MODEL_PATH)

        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "Model loaded successfully."
    )

    print()
    print(
        "Model input shape :",
        model.input_shape
    )

    print(
        "Model output shape:",
        model.output_shape
    )

    # --------------------------------------------------------
    # Get image-mask pairs
    # --------------------------------------------------------

    pairs = get_image_mask_pairs()

    if len(pairs) == 0:

        print()
        print(
            "ERROR: No image-mask pairs found."
        )

        print()
        print(
            "Evaluation stopped."
        )

        return

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print()
    print("Loading test images and masks...")

    images = []
    masks = []

    for image_path, mask_path in pairs:

        image = load_image(
            image_path
        )

        mask = load_mask(
            mask_path
        )

        images.append(image)
        masks.append(mask)

    images = np.array(
        images,
        dtype=np.float32
    )

    masks = np.array(
        masks,
        dtype=np.float32
    )

    print()
    print(
        "Image shape :",
        images.shape
    )

    print(
        "Mask shape  :",
        masks.shape
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print()
    print("Generating predictions...")

    predictions = model.predict(
        images,
        batch_size=16,
        verbose=1
    )

    print()
    print(
        "Raw prediction shape:",
        predictions.shape
    )

    # --------------------------------------------------------
    # Convert probability to binary mask
    # --------------------------------------------------------

    predictions_binary = (
        predictions >= 0.5
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    dice_scores = []
    iou_scores = []

    print()
    print(
        "Calculating Dice and IoU..."
    )

    for i in range(len(masks)):

        true_mask = masks[i]

        predicted_mask = (
            predictions_binary[i]
        )

        dice = dice_score(
            true_mask,
            predicted_mask
        )

        iou = iou_score(
            true_mask,
            predicted_mask
        )

        dice_scores.append(
            dice
        )

        iou_scores.append(
            iou
        )

    # --------------------------------------------------------
    # Mean metrics
    # --------------------------------------------------------

    mean_dice = np.mean(
        dice_scores
    )

    mean_iou = np.mean(
        iou_scores
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Mean Dice : {mean_dice:.4f}"
    )

    print(
        f"Mean Dice (%) : {mean_dice * 100:.2f}%"
    )

    print(
        f"Mean IoU : {mean_iou:.4f}"
    )

    print(
        f"Mean IoU (%) : {mean_iou * 100:.2f}%"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    result_file = os.path.join(
        RESULT_DIR,
        "segmentation_test_results.txt"
    )

    with open(
        result_file,
        "w"
    ) as f:

        f.write(
            "HAIR SEGMENTATION TEST RESULTS\n"
        )

        f.write(
            "====================================\n"
        )

        f.write(
            f"Test images : {len(images)}\n"
        )

        f.write(
            f"Test masks : {len(masks)}\n"
        )

        f.write(
            f"Mean Dice : {mean_dice:.4f}\n"
        )

        f.write(
            f"Mean Dice (%) : "
            f"{mean_dice * 100:.2f}%\n"
        )

        f.write(
            f"Mean IoU : {mean_iou:.4f}\n"
        )

        f.write(
            f"Mean IoU (%) : "
            f"{mean_iou * 100:.2f}%\n"
        )

    print()
    print(
        "Metrics saved to:"
    )

    print(
        result_file
    )

    print()
    print(
        "SEGMENTATION TEST EVALUATION COMPLETE"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()