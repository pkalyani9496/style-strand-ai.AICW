import os
import hashlib
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\CH KAVYA\OneDrive\Desktop\Hair Type Prediction"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "hair_disease"
)

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# CALCULATE IMAGE HASH
# ============================================================

def get_file_hash(filepath):
    """
    Calculate SHA-256 hash of an image file.
    Identical files will have identical hashes.
    """

    sha256 = hashlib.sha256()

    try:
        with open(filepath, "rb") as f:
            while True:
                data = f.read(1024 * 1024)

                if not data:
                    break

                sha256.update(data)

        return sha256.hexdigest()

    except Exception as e:
        print(f"Could not read: {filepath}")
        print(f"Error: {e}")
        return None


# ============================================================
# COLLECT IMAGES
# ============================================================

print("=" * 70)
print("HAIR DISEASE DATASET DUPLICATE CHECK")
print("=" * 70)

all_images = []

for split in SPLITS:

    split_path = os.path.join(DATASET_DIR, split)

    print(f"\nScanning: {split}")

    if not os.path.exists(split_path):
        print(f"ERROR: Folder not found: {split_path}")
        continue

    for root, dirs, files in os.walk(split_path):

        for filename in files:

            extension = os.path.splitext(filename)[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            filepath = os.path.join(root, filename)

            # Class name = folder directly inside train/val/test
            class_name = os.path.basename(root)

            all_images.append({
                "path": filepath,
                "split": split,
                "class": class_name,
                "filename": filename
            })

    split_count = sum(1 for x in all_images if x["split"] == split)

    print(f"Images found: {split_count}")


print("\n" + "=" * 70)
print(f"TOTAL IMAGES FOUND: {len(all_images)}")
print("=" * 70)


# ============================================================
# CALCULATE HASHES
# ============================================================

print("\nCalculating SHA-256 hashes...")
print("Please wait...")

hash_groups = defaultdict(list)

for i, image in enumerate(all_images, start=1):

    file_hash = get_file_hash(image["path"])

    if file_hash is not None:
        hash_groups[file_hash].append(image)

    if i % 500 == 0:
        print(f"Processed {i}/{len(all_images)} images")


# ============================================================
# FIND EXACT DUPLICATES
# ============================================================

duplicate_groups = []

for file_hash, images in hash_groups.items():

    if len(images) > 1:
        duplicate_groups.append((file_hash, images))


# ============================================================
# PRINT DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

print(f"\nUnique image files : {len(hash_groups)}")
print(f"Duplicate groups   : {len(duplicate_groups)}")


if len(duplicate_groups) == 0:

    print("\n✅ NO EXACT DUPLICATES FOUND!")
    print("\nTrain, validation and test sets contain no identical")
    print("image files according to SHA-256 hashing.")

else:

    print("\n⚠️ DUPLICATES FOUND!")

    cross_split_duplicates = 0

    for group_number, (file_hash, images) in enumerate(
        duplicate_groups,
        start=1
    ):

        splits_present = set(
            image["split"]
            for image in images
        )

        # Only interested in duplicates across different splits
        if len(splits_present) > 1:

            cross_split_duplicates += 1

            print("\n" + "-" * 70)
            print(f"DUPLICATE GROUP {group_number}")
            print("-" * 70)

            print(f"Hash: {file_hash}")

            for image in images:

                print(
                    f"\nSplit : {image['split']}"
                    f"\nClass : {image['class']}"
                    f"\nFile  : {image['filename']}"
                    f"\nPath  : {image['path']}"
                )

    print("\n" + "=" * 70)
    print(
        f"Cross-split duplicate groups: "
        f"{cross_split_duplicates}"
    )
    print("=" * 70)


# ============================================================
# SPLIT-PAIR ANALYSIS
# ============================================================

def get_hashes_for_split(split_name):

    hashes = {}

    for file_hash, images in hash_groups.items():

        for image in images:

            if image["split"] == split_name:

                hashes.setdefault(file_hash, []).append(image)

    return hashes


train_hashes = get_hashes_for_split("train")
val_hashes = get_hashes_for_split("val")
test_hashes = get_hashes_for_split("test")


train_val = set(train_hashes) & set(val_hashes)
train_test = set(train_hashes) & set(test_hashes)
val_test = set(val_hashes) & set(test_hashes)


print("\n" + "=" * 70)
print("CROSS-SPLIT DUPLICATE SUMMARY")
print("=" * 70)

print(f"\nTrain ↔ Validation : {len(train_val)}")
print(f"Train ↔ Test       : {len(train_test)}")
print(f"Validation ↔ Test  : {len(val_test)}")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

total_cross_split = (
    len(train_val)
    + len(train_test)
    + len(val_test)
)

if total_cross_split == 0:

    print("""
✅ DATASET SPLIT LOOKS CLEAN

No exact duplicate image files were found between:
    Train ↔ Validation
    Train ↔ Test
    Validation ↔ Test

The 99.92% test accuracy is therefore NOT explained
by exact duplicate files across the dataset splits.

NOTE:
This check detects exact file duplicates using SHA-256.
It does not detect visually identical images that were
resized, cropped, compressed, rotated, or otherwise modified.
""")

else:

    print("""
⚠️ CROSS-SPLIT DUPLICATES FOUND

There are identical image files appearing in more than
one dataset split.

This can cause data leakage and may make the test accuracy
appear higher than the model's true generalization ability.

Review the duplicate groups printed above before reporting
the 99.92% accuracy as the final real-world performance.
""")


print("=" * 70)
print("DUPLICATE CHECK COMPLETED")
print("=" * 70)