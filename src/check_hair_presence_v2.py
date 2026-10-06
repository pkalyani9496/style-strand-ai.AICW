import os
import random
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "datasets/hair_presence_prepared_v2"

CLASSES = ["Hair", "NonHair"]

SAMPLES_PER_CLASS = 10


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    images = []

    for file in os.listdir(folder):

        if file.lower().endswith(extensions):

            images.append(
                os.path.join(folder, file)
            )

    return images


# ============================================================
# DISPLAY SAMPLES
# ============================================================

fig, axes = plt.subplots(
    2,
    SAMPLES_PER_CLASS,
    figsize=(20, 8)
)


for row, class_name in enumerate(CLASSES):

    folder = os.path.join(
        DATASET_DIR,
        "train",
        class_name
    )

    images = get_images(folder)

    samples = random.sample(
        images,
        min(SAMPLES_PER_CLASS, len(images))
    )

    for col, image_path in enumerate(samples):

        image = Image.open(
            image_path
        ).convert("RGB")

        axes[row, col].imshow(image)

        axes[row, col].set_title(
            class_name
        )

        axes[row, col].axis("off")


plt.tight_layout()

plt.show()


print("\n" + "=" * 60)
print("HAIR PRESENCE V2 VISUAL CHECK")
print("=" * 60)

for class_name in CLASSES:

    folder = os.path.join(
        DATASET_DIR,
        "train",
        class_name
    )

    images = get_images(folder)

    print(
        f"{class_name}: {len(images)} images"
    )

print("\nVisual inspection completed.")