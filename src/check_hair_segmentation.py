from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# HAIR SEGMENTATION CONTACT SHEET CHECKER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "hair_segmentation_prepared"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "hair_segmentation"
)

OUTPUT_FILE = (
    RESULTS_ROOT
    / "segmentation_check.png"
)


# ============================================================
# SETTINGS
# ============================================================

SPLIT = "train"

NUMBER_OF_SAMPLES = 5

SEED = 100


# ============================================================
# GET IMAGES
# ============================================================

def get_images():

    image_folder = (
        DATASET_ROOT
        / SPLIT
        / "images"
    )

    if not image_folder.exists():
        return []

    return sorted(
        [
            file
            for file in image_folder.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in [".jpg", ".jpeg", ".png"]
        ]
    )


# ============================================================
# NORMALIZE NAME
# ============================================================

def normalize_name(name):

    name = name.lower()

    name = Path(name).stem

    suffixes = [
        "-org",
        "_org",
        "-gt",
        "_gt",
        "-mask",
        "_mask",
        "-seg",
        "_seg"
    ]

    changed = True

    while changed:

        changed = False

        for suffix in suffixes:

            if name.endswith(suffix):

                name = name[:-len(suffix)]

                changed = True

                break

    name = name.replace("(1)", "")
    name = name.replace(" ", "")

    return name


# ============================================================
# FIND MASK
# ============================================================

def find_mask(image_path):

    mask_folder = (
        DATASET_ROOT
        / SPLIT
        / "masks"
    )

    image_key = normalize_name(
        image_path.name
    )

    for mask_path in mask_folder.iterdir():

        if not mask_path.is_file():
            continue

        if mask_path.suffix.lower() != ".pbm":
            continue

        if "(1)" in mask_path.stem:
            continue

        mask_key = normalize_name(
            mask_path.name
        )

        if image_key == mask_key:

            return mask_path

    return None


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    return Image.open(
        image_path
    ).convert("RGB")


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(mask_path):

    return Image.open(
        mask_path
    ).convert("L")


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(image, mask):

    image_array = np.array(
        image
    )

    mask_array = np.array(
        mask
    )

    # Make sure mask has same size as image
    if mask_array.shape[:2] != image_array.shape[:2]:

        mask = mask.resize(
            (
                image_array.shape[1],
                image_array.shape[0]
            ),
            Image.Resampling.NEAREST
        )

        mask_array = np.array(
            mask
        )

    # Binary mask
    binary_mask = mask_array > 0

    # Copy original image
    overlay = image_array.copy()

    # Highlight hair region
    highlight = np.zeros_like(
        overlay
    )

    highlight[:, :, 0] = 255

    # Blend highlighted region
    overlay[binary_mask] = (
        overlay[binary_mask] * 0.5
        + highlight[binary_mask] * 0.5
    ).astype(np.uint8)

    return overlay


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("HAIR SEGMENTATION CONTACT SHEET CHECK")
    print("=" * 65)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        print("\nERROR: Dataset folder not found:")
        print(DATASET_ROOT)

        return

    # --------------------------------------------------------
    # Get images
    # --------------------------------------------------------

    images = get_images()

    print(
        f"\nImages available in {SPLIT}: {len(images)}"
    )

    if len(images) == 0:

        print("\nERROR: No images found.")

        return

    # --------------------------------------------------------
    # Select samples
    # --------------------------------------------------------

    random.seed(SEED)

    sample_count = min(
        NUMBER_OF_SAMPLES,
        len(images)
    )

    selected_images = random.sample(
        images,
        sample_count
    )

    print(
        f"Selected samples: {sample_count}"
    )

    # --------------------------------------------------------
    # Load samples
    # --------------------------------------------------------

    samples = []

    for image_path in selected_images:

        mask_path = find_mask(
            image_path
        )

        print("\nImage:")
        print(
            image_path.name
        )

        if mask_path is None:

            print(
                "ERROR: Mask not found!"
            )

            continue

        print("Mask:")
        print(
            mask_path.name
        )

        image = load_image(
            image_path
        )

        mask = load_mask(
            mask_path
        )

        mask_array = np.array(
            mask
        )

        hair_pixels = np.sum(
            mask_array > 0
        )

        total_pixels = mask_array.size

        hair_percentage = (
            hair_pixels
            / total_pixels
            * 100
        )

        print(
            "Image size:",
            image.size
        )

        print(
            "Mask size:",
            mask.size
        )

        print(
            "Mask values:",
            np.unique(mask_array)
        )

        print(
            f"Hair mask area: {hair_percentage:.2f}%"
        )

        overlay = create_overlay(
            image,
            mask
        )

        samples.append(
            {
                "name": image_path.name,
                "image": image,
                "mask": mask,
                "overlay": overlay
            }
        )

    # --------------------------------------------------------
    # Check valid samples
    # --------------------------------------------------------

    if len(samples) == 0:

        print(
            "\nERROR: No valid samples available."
        )

        return

    # --------------------------------------------------------
    # Create results folder
    # --------------------------------------------------------

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create contact sheet
    # --------------------------------------------------------

    print("\nCreating contact sheet...")

    figure, axes = plt.subplots(
        len(samples),
        3,
        figsize=(15, 5 * len(samples))
    )

    # If only one row, convert axes to 2D
    if len(samples) == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row, sample in enumerate(samples):

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        axes[row, 0].imshow(
            sample["image"]
        )

        axes[row, 0].set_title(
            f"Original\n{sample['name']}"
        )

        axes[row, 0].axis("off")

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        axes[row, 1].imshow(
            sample["mask"],
            cmap="gray"
        )

        axes[row, 1].set_title(
            "Hair Mask"
        )

        axes[row, 1].axis("off")

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        axes[row, 2].imshow(
            sample["overlay"]
        )

        axes[row, 2].set_title(
            "Hair Mask Overlay"
        )

        axes[row, 2].axis("off")

    figure.suptitle(
        "Figaro1k Hair Segmentation Dataset Check",
        fontsize=18
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    figure.savefig(
        OUTPUT_FILE,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print("\n" + "=" * 65)
    print("CONTACT SHEET CREATED")
    print("=" * 65)

    print("\nSaved to:")
    print(
        OUTPUT_FILE
    )

    print("\nOpen this file to inspect:")
    print(
        "results\\hair_segmentation\\segmentation_check.png"
    )

    print("\n" + "=" * 65)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()