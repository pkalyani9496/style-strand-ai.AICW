import os
import shutil
import random


# ============================================================
# PATHS
# ============================================================

HAIR_SOURCE = "datasets/hair_type"

NONHAIR_SOURCE = "datasets/hair_segmentation/Patch1k/NonHair"

OUTPUT_DIR = "datasets/hair_presence_prepared_v2"


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


random.seed(SEED)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "validation", "test"]:

    os.makedirs(
        os.path.join(OUTPUT_DIR, split, "Hair"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(OUTPUT_DIR, split, "NonHair"),
        exist_ok=True
    )


# ============================================================
# GET IMAGE FILES
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


def get_images(folder):

    images = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(IMAGE_EXTENSIONS):

                images.append(
                    os.path.join(root, file)
                )

    return images


# ============================================================
# COLLECT HAIR IMAGES
# ============================================================

print("\nCollecting REAL hair images...")

hair_images = get_images(HAIR_SOURCE)

print(
    f"Total Hair images found: {len(hair_images)}"
)


# ============================================================
# COLLECT NON-HAIR IMAGES
# ============================================================

print("\nCollecting NonHair images...")

nonhair_images = get_images(NONHAIR_SOURCE)

print(
    f"Total NonHair images found: {len(nonhair_images)}"
)


# ============================================================
# CHECK DATA
# ============================================================

if len(hair_images) == 0:

    raise ValueError(
        "No Hair images found!"
    )


if len(nonhair_images) == 0:

    raise ValueError(
        "No NonHair images found!"
    )


# ============================================================
# BALANCE THE DATASET
# ============================================================

# Use the same number of images from both classes.

number_of_images = min(
    len(hair_images),
    len(nonhair_images)
)

hair_images = random.sample(
    hair_images,
    number_of_images
)

nonhair_images = random.sample(
    nonhair_images,
    number_of_images
)


print("\nBalanced dataset:")
print(f"Hair     : {len(hair_images)}")
print(f"NonHair  : {len(nonhair_images)}")


# ============================================================
# SPLIT FUNCTION
# ============================================================

def split_images(images):

    random.shuffle(images)

    total = len(images)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train_images = images[:train_end]

    validation_images = images[
        train_end:val_end
    ]

    test_images = images[
        val_end:
    ]

    return (
        train_images,
        validation_images,
        test_images
    )


# ============================================================
# SPLIT HAIR
# ============================================================

(
    hair_train,
    hair_validation,
    hair_test
) = split_images(hair_images)


# ============================================================
# SPLIT NON-HAIR
# ============================================================

(
    nonhair_train,
    nonhair_validation,
    nonhair_test
) = split_images(nonhair_images)


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_images(
    images,
    destination,
    class_name
):

    for index, source_path in enumerate(images):

        extension = os.path.splitext(
            source_path
        )[1]

        new_filename = (
            f"{class_name}_{index:05d}"
            f"{extension}"
        )

        destination_path = os.path.join(
            destination,
            class_name,
            new_filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )


# ============================================================
# COPY TRAIN DATA
# ============================================================

print("\nCopying training images...")

copy_images(
    hair_train,
    OUTPUT_DIR + "/train",
    "Hair"
)

copy_images(
    nonhair_train,
    OUTPUT_DIR + "/train",
    "NonHair"
)


# ============================================================
# COPY VALIDATION DATA
# ============================================================

print("Copying validation images...")

copy_images(
    hair_validation,
    OUTPUT_DIR + "/validation",
    "Hair"
)

copy_images(
    nonhair_validation,
    OUTPUT_DIR + "/validation",
    "NonHair"
)


# ============================================================
# COPY TEST DATA
# ============================================================

print("Copying test images...")

copy_images(
    hair_test,
    OUTPUT_DIR + "/test",
    "Hair"
)

copy_images(
    nonhair_test,
    OUTPUT_DIR + "/test",
    "NonHair"
)


# ============================================================
# PRINT FINAL COUNTS
# ============================================================

print("\n" + "=" * 60)
print("HAIR PRESENCE V2 DATASET PREPARATION COMPLETE")
print("=" * 60)

print("\nTRAIN:")
print(f"Hair     : {len(hair_train)}")
print(f"NonHair  : {len(nonhair_train)}")
print(
    f"Total    : "
    f"{len(hair_train) + len(nonhair_train)}"
)

print("\nVALIDATION:")
print(f"Hair     : {len(hair_validation)}")
print(f"NonHair  : {len(nonhair_validation)}")
print(
    f"Total    : "
    f"{len(hair_validation) + len(nonhair_validation)}"
)

print("\nTEST:")
print(f"Hair     : {len(hair_test)}")
print(f"NonHair  : {len(nonhair_test)}")
print(
    f"Total    : "
    f"{len(hair_test) + len(nonhair_test)}"
)

print("\nDataset location:")
print(
    os.path.abspath(OUTPUT_DIR)
)

print("\nDone!")