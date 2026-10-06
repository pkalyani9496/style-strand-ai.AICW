from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(".")


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# IMAGE DATASET INSPECTION
# ============================================================

def inspect_image_dataset(folder):

    print("\n" + "=" * 70)
    print(f"IMAGE DATASET: {folder}")
    print("=" * 70)

    # Check folder
    if not folder.exists():

        print("❌ Folder not found")
        return

    # --------------------------------------------------------
    # Find valid images
    # --------------------------------------------------------

    image_files = [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and "__MACOSX" not in p.parts
        and not p.name.startswith("._")
    ]

    print(f"Valid images: {len(image_files)}")

    # --------------------------------------------------------
    # Find MacOS metadata images
    # --------------------------------------------------------

    all_image_files = [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    unwanted_files = [
        p for p in all_image_files
        if "__MACOSX" in p.parts
        or p.name.startswith("._")
    ]

    print(f"MacOS metadata images: {len(unwanted_files)}")

    # --------------------------------------------------------
    # No images
    # --------------------------------------------------------

    if len(image_files) == 0:

        print("❌ No valid images found")
        return

    # --------------------------------------------------------
    # Folder structure
    # --------------------------------------------------------

    print("\nFolder structure:")

    directories = sorted(
        {
            p.parent.relative_to(folder)
            for p in image_files
        }
    )

    for directory in directories[:100]:

        print(f"  {directory}")

    # --------------------------------------------------------
    # Count images by immediate parent folder
    # --------------------------------------------------------

    parent_counts = Counter(
        p.parent.name
        for p in image_files
    )

    print("\nImages by immediate parent folder:")

    for name, count in parent_counts.most_common():

        print(f"  {name}: {count}")

    # --------------------------------------------------------
    # Check corrupted images
    # --------------------------------------------------------

    corrupted = []

    for image_path in image_files:

        try:

            with Image.open(image_path) as img:

                img.verify()

        except Exception:

            corrupted.append(str(image_path))

    print(f"\nCorrupted valid images: {len(corrupted)}")

    if corrupted:

        print("\nFirst corrupted images:")

        for file in corrupted[:10]:

            print(file)

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    sizes = Counter()

    for image_path in image_files[:100]:

        try:

            with Image.open(image_path) as img:

                sizes[img.size] += 1

        except Exception:

            pass

    print("\nSample image dimensions:")

    for size, count in sizes.most_common(10):

        print(f"  {size}: {count}")


# ============================================================
# CSV DATASET INSPECTION
# ============================================================

def inspect_csv_dataset(folder):

    print("\n" + "=" * 70)
    print(f"CSV DATASET FOLDER: {folder}")
    print("=" * 70)

    # Check folder
    if not folder.exists():

        print("❌ CSV folder not found")
        return

    # --------------------------------------------------------
    # Find CSV files
    # --------------------------------------------------------

    csv_files = list(folder.glob("*.csv"))

    print(f"CSV files found: {len(csv_files)}")

    # --------------------------------------------------------
    # If CSV not found
    # --------------------------------------------------------

    if len(csv_files) == 0:

        print("❌ No CSV file found")

        print("\nFiles actually inside this folder:")

        for file in folder.iterdir():

            print(f"  {file.name}")

        return

    # --------------------------------------------------------
    # Inspect each CSV file
    # --------------------------------------------------------

    for csv_file in csv_files:

        print("\n" + "-" * 70)
        print(f"CSV FILE: {csv_file.name}")
        print("-" * 70)

        try:

            df = pd.read_csv(csv_file)

        except Exception as e:

            print("❌ Error reading CSV:")
            print(e)

            continue

        # ----------------------------------------------------
        # Shape
        # ----------------------------------------------------

        print("\nShape:")

        print(df.shape)

        # ----------------------------------------------------
        # Columns
        # ----------------------------------------------------

        print("\nColumns:")

        for column in df.columns:

            print(f"  {column}")

        # ----------------------------------------------------
        # Data types
        # ----------------------------------------------------

        print("\nData types:")

        print(df.dtypes)

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        print("\nMissing values:")

        print(df.isnull().sum())

        # ----------------------------------------------------
        # Duplicate rows
        # ----------------------------------------------------

        print("\nDuplicate rows:")

        print(df.duplicated().sum())

        # ----------------------------------------------------
        # First 5 rows
        # ----------------------------------------------------

        print("\nFirst 5 rows:")

        print(df.head())

        # ----------------------------------------------------
        # Numerical summary
        # ----------------------------------------------------

        print("\nNumerical summary:")

        print(df.describe(include="all"))


# ============================================================
# DATASET INSPECTION
# ============================================================

print("\n")
print("=" * 70)
print("HAIR AI PROJECT - DATASET INSPECTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. HAIR TYPE DATASET
# ------------------------------------------------------------

inspect_image_dataset(
    ROOT / "datasets" / "hair_type"
)


# ------------------------------------------------------------
# 2. HAIR DISEASE DATASET
# ------------------------------------------------------------

inspect_image_dataset(
    ROOT / "datasets" / "hair_disease"
)


# ------------------------------------------------------------
# 3. HAIR SEGMENTATION DATASET
# ------------------------------------------------------------

inspect_image_dataset(
    ROOT / "datasets" / "hair_segmentation"
)


# ------------------------------------------------------------
# 4. HAIR LOSS CSV DATASET
# ------------------------------------------------------------

inspect_csv_dataset(
    Path(r"C:\Users\CH KAVYA\OneDrive\Desktop\Hair Type Prediction\hair_loss")
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("✅ INSPECTION COMPLETED")
print("=" * 70)