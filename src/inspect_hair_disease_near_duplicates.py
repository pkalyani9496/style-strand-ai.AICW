import os
from PIL import Image, ImageDraw, ImageFont
import imagehash

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\CH KAVYA\OneDrive\Desktop\Hair Type Prediction"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "hair_disease"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "hair_disease",
    "near_duplicate_inspection"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

THRESHOLD = 5


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_images(split):

    split_path = os.path.join(
        DATASET_DIR,
        split
    )

    images = []

    for root, dirs, files in os.walk(split_path):

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            filepath = os.path.join(
                root,
                filename
            )

            class_name = os.path.basename(root)

            images.append({
                "path": filepath,
                "split": split,
                "class": class_name,
                "filename": filename
            })

    return images


# ============================================================
# CALCULATE PERCEPTUAL HASH
# ============================================================

def calculate_hash(images):

    print(
        f"\nCalculating hashes for "
        f"{len(images)} images..."
    )

    for i, image in enumerate(images, start=1):

        try:

            with Image.open(image["path"]) as img:

                img = img.convert("RGB")

                image["hash"] = imagehash.phash(img)

        except Exception as e:

            image["hash"] = None

            print(
                f"Could not process: "
                f"{image['path']}"
            )

        if i % 500 == 0:
            print(
                f"Processed "
                f"{i}/{len(images)}"
            )


# ============================================================
# FIND NEAR DUPLICATES
# ============================================================

def find_matches(train_images, test_images):

    matches = []

    print("\nComparing Train ↔ Test...")

    for train_image in train_images:

        if train_image["hash"] is None:
            continue

        for test_image in test_images:

            if test_image["hash"] is None:
                continue

            distance = (
                train_image["hash"]
                - test_image["hash"]
            )

            if distance <= THRESHOLD:

                matches.append({
                    "distance": distance,
                    "train": train_image,
                    "test": test_image
                })

    matches.sort(
        key=lambda x: x["distance"]
    )

    return matches


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(matches):

    print(
        f"\nCreating contact sheets for "
        f"{len(matches)} near-duplicate pairs..."
    )

    # Image dimensions
    image_width = 300
    image_height = 250

    # Number of pairs per sheet
    pairs_per_sheet = 10

    total_sheets = (
        len(matches) + pairs_per_sheet - 1
    ) // pairs_per_sheet

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            16
        )
    except:
        font = ImageFont.load_default()

    for sheet_number in range(total_sheets):

        start = (
            sheet_number
            * pairs_per_sheet
        )

        end = min(
            start + pairs_per_sheet,
            len(matches)
        )

        current_matches = matches[start:end]

        sheet_width = image_width * 2
        sheet_height = (
            image_height
            * len(current_matches)
        )

        sheet = Image.new(
            "RGB",
            (
                sheet_width,
                sheet_height
            ),
            "white"
        )

        draw = ImageDraw.Draw(sheet)

        for row, match in enumerate(
            current_matches
        ):

            train_path = match["train"]["path"]
            test_path = match["test"]["path"]

            try:

                train_img = Image.open(
                    train_path
                ).convert("RGB")

                test_img = Image.open(
                    test_path
                ).convert("RGB")

                train_img.thumbnail(
                    (
                        image_width,
                        image_height - 50
                    )
                )

                test_img.thumbnail(
                    (
                        image_width,
                        image_height - 50
                    )
                )

                y = row * image_height

                # Paste images
                sheet.paste(
                    train_img,
                    (
                        (image_width - train_img.width) // 2,
                        y
                    )
                )

                sheet.paste(
                    test_img,
                    (
                        image_width
                        + (image_width - test_img.width) // 2,
                        y
                    )
                )

                # Text
                train_text = (
                    f"TRAIN | "
                    f"{match['train']['class']}\n"
                    f"{match['train']['filename']}"
                )

                test_text = (
                    f"TEST | "
                    f"{match['test']['class']}\n"
                    f"{match['test']['filename']}"
                )

                draw.text(
                    (
                        5,
                        y + image_height - 45
                    ),
                    train_text,
                    fill="black",
                    font=font
                )

                draw.text(
                    (
                        image_width + 5,
                        y + image_height - 45
                    ),
                    test_text,
                    fill="black",
                    font=font
                )

                # Hash distance
                draw.text(
                    (
                        image_width - 50,
                        y + 5
                    ),
                    f"D={match['distance']}",
                    fill="red",
                    font=font
                )

            except Exception as e:

                print(
                    f"Could not create preview for:"
                    f"\n{train_path}"
                    f"\n{test_path}"
                    f"\nError: {e}"
                )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"train_test_sheet_{sheet_number + 1}.jpg"
        )

        sheet.save(
            output_path,
            quality=95
        )

        print(
            f"Saved: {output_path}"
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("HAIR DISEASE NEAR-DUPLICATE VISUAL INSPECTION")
print("=" * 70)

print("\nLoading Train images...")

train_images = collect_images("train")

print(
    f"Train images: "
    f"{len(train_images)}"
)

print("\nLoading Test images...")

test_images = collect_images("test")

print(
    f"Test images: "
    f"{len(test_images)}"
)

# Calculate hashes
calculate_hash(train_images)
calculate_hash(test_images)

# Find near duplicates
matches = find_matches(
    train_images,
    test_images
)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"\nTrain ↔ Test near-duplicate pairs: "
    f"{len(matches)}"
)

# Create visual sheets
if len(matches) > 0:

    create_contact_sheet(matches)

else:

    print(
        "\nNo near-duplicate pairs found."
    )

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)

print(
    f"\nContact sheets saved to:"
    f"\n{OUTPUT_DIR}"
)