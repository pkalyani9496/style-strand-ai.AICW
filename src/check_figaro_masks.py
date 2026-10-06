from pathlib import Path

original_dir = Path(
    "datasets/hair_segmentation/Figaro1k/Figaro1k/Original/Training"
)

gt_dir = Path(
    "datasets/hair_segmentation/Figaro1k/Figaro1k/GT/Training"
)

# Original images
original_images = {
    p.stem.replace("-org", "")
    for p in original_dir.glob("*.jpg")
}

# Normal GT masks
# Ignore duplicate files containing (1)
gt_masks = {
    p.stem.replace("-gt", "")
    for p in gt_dir.glob("*.pbm")
    if "(1)" not in p.stem
}

# Duplicate masks
duplicate_masks = [
    p.name for p in gt_dir.glob("*(1).pbm")
]

# Compare
missing_masks = sorted(original_images - gt_masks)
extra_masks = sorted(gt_masks - original_images)

print("=" * 60)
print("FIGARO1K IMAGE - MASK CHECK")
print("=" * 60)

print(f"Original training images : {len(original_images)}")
print(f"GT training masks        : {len(gt_masks)}")
print(f"Duplicate masks          : {len(duplicate_masks)}")
print(f"Missing masks            : {len(missing_masks)}")
print(f"Extra masks              : {len(extra_masks)}")

print("\nDuplicate mask count:", len(duplicate_masks))

print("\nMissing masks:")
if missing_masks:
    for name in missing_masks:
        print("  ", name)
else:
    print("  None")

print("\nExtra masks:")
if extra_masks:
    for name in extra_masks:
        print("  ", name)
else:
    print("  None")

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)