import os
from PIL import Image
import imagehash

# ============================================================
# PATH
# ============================================================

BASE_DIR = r"C:\Users\CH KAVYA\OneDrive\Desktop\Hair Type Prediction"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "hair_disease"
)

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# Smaller threshold = more similar
# 0 = identical visual hash
# 1-4 = very similar
# 5-8 = potentially similar
THRESHOLD = 5


# ============================================================
# COLLECT IMAGES
# ============================================================

print("=" * 70)
print("HAIR DISEASE NEAR-DUPLICATE CHECK")
print("=" * 70)

images = []

for split in SPLITS:

    split_path = os.path.join(DATASET_DIR, split)

    print(f"\nScanning {split}...")

    for root, dirs, files in os.walk(split_path):

        for filename in files:

            ext = os.path.splitext(filename)[1].lower()

            if ext not in IMAGE_EXTENSIONS:
                continue

            filepath = os.path.join(root, filename)

            class_name = os.path.basename(root)

            images.append({
                "path": filepath,
                "split": split,
                "class": class_name,
                "filename": filename
            })

print(f"\nTotal images: {len(images)}")


# ============================================================
# CALCULATE PERCEPTUAL HASHES
# ============================================================

print("\nCalculating perceptual hashes...")
print("Please wait...")

for i, image in enumerate(images, start=1):

    try:

        with Image.open(image["path"]) as img:

            img = img.convert("RGB")

            # Perceptual hash
            image["hash"] = imagehash.phash(img)

    except Exception as e:

        image["hash"] = None

        print(
            f"Could not process: "
            f"{image['path']}"
        )

    if i % 500 == 0:
        print(f"Processed {i}/{len(images)}")


# ============================================================
# SPLIT IMAGES
# ============================================================

train_images = [
    x for x in images
    if x["split"] == "train" and x["hash"] is not None
]

val_images = [
    x for x in images
    if x["split"] == "val" and x["hash"] is not None
]

test_images = [
    x for x in images
    if x["split"] == "test" and x["hash"] is not None
]


# ============================================================
# COMPARE FUNCTION
# ============================================================

def compare_sets(set_a, set_b, name):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    matches = []

    for i, image_a in enumerate(set_a):

        for image_b in set_b:

            distance = image_a["hash"] - image_b["hash"]

            if distance <= THRESHOLD:

                matches.append(
                    (
                        distance,
                        image_a,
                        image_b
                    )
                )

    matches.sort(key=lambda x: x[0])

    print(
        f"\nNear-duplicate pairs found: "
        f"{len(matches)}"
    )

    # Show maximum 30 examples
    for number, (
        distance,
        image_a,
        image_b
    ) in enumerate(matches[:30], start=1):

        print("\n" + "-" * 70)

        print(f"Match {number}")
        print(f"Hash distance : {distance}")

        print(
            f"\nImage A"
            f"\nSplit : {image_a['split']}"
            f"\nClass : {image_a['class']}"
            f"\nFile  : {image_a['filename']}"
        )

        print(
            f"\nImage B"
            f"\nSplit : {image_b['split']}"
            f"\nClass : {image_b['class']}"
            f"\nFile  : {image_b['filename']}"
        )

    return matches


# ============================================================
# CROSS-SPLIT COMPARISONS
# ============================================================

train_val_matches = compare_sets(
    train_images,
    val_images,
    "TRAIN ↔ VALIDATION"
)

train_test_matches = compare_sets(
    train_images,
    test_images,
    "TRAIN ↔ TEST"
)

val_test_matches = compare_sets(
    val_images,
    test_images,
    "VALIDATION ↔ TEST"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL NEAR-DUPLICATE SUMMARY")
print("=" * 70)

print(
    f"\nTrain ↔ Validation : "
    f"{len(train_val_matches)}"
)

print(
    f"Train ↔ Test       : "
    f"{len(train_test_matches)}"
)

print(
    f"Validation ↔ Test  : "
    f"{len(val_test_matches)}"
)


total_matches = (
    len(train_val_matches)
    + len(train_test_matches)
    + len(val_test_matches)
)


print(
    f"\nTotal near-duplicate pairs: "
    f"{total_matches}"
)


if total_matches == 0:

    print("""
    
✅ NO NEAR-DUPLICATE PAIRS FOUND

The dataset appears clean based on both:

1. SHA-256 exact duplicate checking
2. Perceptual-hash near-duplicate checking

The 99.92% test accuracy is therefore not obviously
caused by duplicate or highly similar images across
the train, validation and test splits.

""")

else:

    print("""
    
⚠️ NEAR-DUPLICATES FOUND

Some visually similar images exist across dataset splits.

This does not automatically mean data leakage.

We need to inspect the reported pairs and determine
whether they are actually the same underlying image
or simply similar-looking disease images.

""")


print("=" * 70)
print("NEAR-DUPLICATE CHECK COMPLETED")
print("=" * 70)