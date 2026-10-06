from pathlib import Path


# ============================================================
# HAIR COLOUR RECOMMENDATION ENGINE
# ============================================================

COLOR_INFO = {
    "Burgundy": {
        "description": "A deep red-purple shade that gives a bold and stylish look.",
        "suitable_for": ["Curly Hair", "Wavy Hair", "Straight Hair"],
        "intensity": "Bold",
        "maintenance": "Medium"
    },

    "Copper": {
        "description": "A warm orange-red shade that gives a vibrant appearance.",
        "suitable_for": ["Curly Hair", "Wavy Hair", "Straight Hair"],
        "intensity": "Bold",
        "maintenance": "Medium"
    },

    "Chocolate Brown": {
        "description": "A natural-looking warm brown shade.",
        "suitable_for": ["Straight Hair", "Wavy Hair", "Curly Hair"],
        "intensity": "Natural",
        "maintenance": "Low"
    },

    "Ash Brown": {
        "description": "A cool-toned brown shade with a subtle modern appearance.",
        "suitable_for": ["Straight Hair", "Wavy Hair"],
        "intensity": "Subtle",
        "maintenance": "Medium"
    },

    "Honey Blonde": {
        "description": "A warm golden-blonde shade with a bright appearance.",
        "suitable_for": ["Straight Hair", "Wavy Hair"],
        "intensity": "Bright",
        "maintenance": "High"
    },

    "Purple": {
        "description": "A vibrant purple shade for a creative and expressive look.",
        "suitable_for": ["Curly Hair", "Wavy Hair", "Straight Hair"],
        "intensity": "Bold",
        "maintenance": "High"
    },

    "Caramel Brown": {
        "description": "A warm caramel-brown shade that provides a soft highlighted appearance.",
        "suitable_for": ["Curly Hair", "Wavy Hair", "Straight Hair"],
        "intensity": "Natural",
        "maintenance": "Medium"
    },

    "Black": {
        "description": "A deep black shade that provides a classic appearance.",
        "suitable_for": ["Straight Hair", "Wavy Hair", "Curly Hair"],
        "intensity": "Natural",
        "maintenance": "Low"
    }
}


# ============================================================
# RECOMMENDATION RULES
# ============================================================

RECOMMENDATIONS = {

    "Curly Hair": [
        "Burgundy",
        "Copper",
        "Caramel Brown",
        "Chocolate Brown",
        "Purple"
    ],

    "Wavy Hair": [
        "Copper",
        "Caramel Brown",
        "Chocolate Brown",
        "Ash Brown",
        "Burgundy"
    ],

    "Straight Hair": [
        "Ash Brown",
        "Chocolate Brown",
        "Honey Blonde",
        "Burgundy",
        "Caramel Brown"
    ]
}


# ============================================================
# FUNCTION
# ============================================================

def recommend_colors(hair_type):

    hair_type = hair_type.strip()

    if hair_type not in RECOMMENDATIONS:
        print("\nInvalid hair type.")
        print("Choose:")
        print("1. Curly Hair")
        print("2. Wavy Hair")
        print("3. Straight Hair")
        return

    colors = RECOMMENDATIONS[hair_type]

    print("\n" + "=" * 70)
    print("HAIR COLOUR RECOMMENDATION")
    print("=" * 70)

    print(f"\nDetected Hair Type: {hair_type}")

    print("\nRecommended Colours:\n")

    for i, color in enumerate(colors, start=1):

        info = COLOR_INFO[color]

        print(f"{i}. {color}")
        print(f"   Description : {info['description']}")
        print(f"   Intensity   : {info['intensity']}")
        print(f"   Maintenance: {info['maintenance']}")
        print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI HAIR COLOUR RECOMMENDATION")
    print("=" * 70)

    print("\nSelect Hair Type:")
    print("1. Curly Hair")
    print("2. Wavy Hair")
    print("3. Straight Hair")

    choice = input("\nEnter choice (1/2/3): ").strip()

    hair_types = {
        "1": "Curly Hair",
        "2": "Wavy Hair",
        "3": "Straight Hair"
    }

    if choice in hair_types:

        recommend_colors(
            hair_types[choice]
        )

    else:

        print("\nInvalid choice.")