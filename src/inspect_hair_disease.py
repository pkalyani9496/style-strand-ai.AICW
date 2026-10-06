from pathlib import Path
from PIL import Image

DATASET_DIR = Path("datasets/hair_disease")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("=" * 70)
print("HAIR DISEASE DATASET INSPECTION")
print("=" * 70)

if not DATASET_DIR.exists():
    print("ERROR: Dataset folder not found.")
    print(DATASET_DIR)
    exit()

print("\nDataset location:")
print(DATASET_DIR.resolve())

print("\n" + "=" * 70)
print("FOLDER STRUCTURE")
print("=" * 70)

for item in DATASET_DIR.iterdir():
    if item.is_dir():
        print("\nFolder:", item.name)

        class_folders = [
            x for x in item.iterdir()
            if x.is_dir()
        ]

        if class_folders:
            for class_dir in sorted(class_folders):
                images = [
                    p for p in class_dir.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in IMAGE_EXTENSIONS
                    and not p.name.startswith("._")
                ]

                print(f"  {class_dir.name}: {len(images)} images")

print("\n" + "=" * 70)
print("IMAGE VALIDATION")
print("=" * 70)

total_images = 0
valid_images = 0
corrupt_images = 0

for image_path in DATASET_DIR.rglob("*"):
    if (
        image_path.is_file()
        and image_path.suffix.lower() in IMAGE_EXTENSIONS
        and not image_path.name.startswith("._")
    ):
        total_images += 1

        try:
            with Image.open(image_path) as img:
                img.verify()

            valid_images += 1

        except Exception:
            corrupt_images += 1
            print("Corrupt:", image_path)

print("\nTotal images   :", total_images)
print("Valid images   :", valid_images)
print("Corrupt images :", corrupt_images)

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)