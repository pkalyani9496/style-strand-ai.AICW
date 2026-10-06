from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random

DATASET_DIR = Path("datasets/hair_disease/train")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Get all disease classes
classes = sorted([
    folder.name
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
])

print("=" * 70)
print("HAIR DISEASE IMAGE VISUAL INSPECTION")
print("=" * 70)

print("\nClasses found:")
for i, class_name in enumerate(classes):
    print(f"{i}: {class_name}")

# Select 1 random image from each class
selected_images = []

for class_name in classes:
    class_dir = DATASET_DIR / class_name

    image_files = [
        p for p in class_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.name.startswith("._")
    ]

    selected_image = random.choice(image_files)
    selected_images.append((class_name, selected_image))

# Create figure
fig, axes = plt.subplots(2, 5, figsize=(18, 8))

for ax, (class_name, image_path) in zip(axes.ravel(), selected_images):

    image = Image.open(image_path)

    ax.imshow(image)
    ax.set_title(class_name, fontsize=10)
    ax.axis("off")

plt.suptitle(
    "Hair Disease Dataset - Sample Images",
    fontsize=16
)

plt.tight_layout()
plt.show()