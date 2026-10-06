from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random


DATASET_DIR = Path("datasets/hair_type")

CLASSES = [
    "Curly Hair",
    "Straight Hair",
    "Wavy Hair"
]


fig, axes = plt.subplots(3, 5, figsize=(15, 10))

for row, class_name in enumerate(CLASSES):

    class_dir = DATASET_DIR / class_name

    image_files = [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }
    ]

    selected_images = random.sample(
        image_files,
        min(5, len(image_files))
    )

    for col, image_path in enumerate(selected_images):

        image = Image.open(image_path)

        axes[row, col].imshow(image)

        axes[row, col].set_title(class_name)

        axes[row, col].axis("off")


plt.tight_layout()
plt.show()