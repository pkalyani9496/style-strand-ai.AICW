import os
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "hair_condition_gate_prepared_v2"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "hair_condition_gate"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "hair_condition_gate"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("=" * 70)
print("HAIR CONDITION GATE V2 - TRAINING")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDataset:")
print(DATASET_DIR)

print("\nClasses:")
print("0 = Disease")
print("1 = Normal / No Visible Condition")


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"


# ============================================================
# CHECK DATASET
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATASET")
print("=" * 70)

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_DIR}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation folder not found:\n{VAL_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test folder not found:\n{TEST_DIR}"
    )

print("Dataset folders found successfully.")


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)


train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\nDetected classes:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)

test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# BUILD MOBILENETV2
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MOBILENETV2")
print("=" * 70)


base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# ============================================================
# PHASE 1 - FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# MODEL INPUT
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

x = data_augmentation(inputs)


# ============================================================
# MOBILENETV2 PREPROCESSING
# ============================================================

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)


# ============================================================
# MOBILENETV2 BASE
# ============================================================

x = base_model(
    x,
    training=False
)


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


# ============================================================
# CREATE MODEL
# ============================================================

model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# PHASE 1 COMPILATION
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# MODEL PATHS
# ============================================================

best_model_path = (
    MODEL_DIR
    / "best_hair_condition_gate_v2.keras"
)

final_model_path = (
    MODEL_DIR
    / "hair_condition_gate_v2.keras"
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    best_model_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=False,
    verbose=1
)


# ============================================================
# PHASE 1
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1: TRANSFER LEARNING")
print("=" * 70)

print("\nTraining with frozen MobileNetV2 base...")


history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        checkpoint,
        reduce_lr,
        early_stopping
    ]
)


# ============================================================
# PHASE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2: FINE TUNING")
print("=" * 70)

print("\nUnfreezing last 100 MobileNetV2 layers...")


base_model.trainable = True


# Freeze all except last 100 layers

for layer in base_model.layers[:-100]:
    layer.trainable = False


# Keep BatchNormalization layers frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False


# ============================================================
# PHASE 2 COMPILATION
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[
        checkpoint,
        reduce_lr,
        early_stopping
    ]
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


if not best_model_path.exists():

    raise FileNotFoundError(
        "Best model was not created."
    )


best_model = tf.keras.models.load_model(
    best_model_path
)


print("Best model loaded successfully.")


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)


test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)


print("\n" + "=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

best_model.save(
    final_model_path
)


print("\nFinal model saved to:")

print(final_model_path)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = (
    MODEL_DIR
    / "class_names.json"
)


with open(
    class_names_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )


print("\nClass names saved to:")

print(class_names_path)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_data = {

    "phase1": {
        key: [
            float(v)
            for v in values
        ]

        for key, values
        in history1.history.items()
    },

    "phase2": {
        key: [
            float(v)
            for v in values
        ]

        for key, values
        in history2.history.items()
    }
}


history_path = (
    RESULTS_DIR
    / "training_history.json"
)


with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history_data,
        f,
        indent=4
    )


print("\nTraining history saved to:")

print(history_path)


# ============================================================
# TRAINING PLOTS
# ============================================================

import matplotlib.pyplot as plt


train_accuracy = (
    history1.history["accuracy"]
    +
    history2.history["accuracy"]
)


val_accuracy = (
    history1.history["val_accuracy"]
    +
    history2.history["val_accuracy"]
)


train_loss = (
    history1.history["loss"]
    +
    history2.history["loss"]
)


val_loss = (
    history1.history["val_loss"]
    +
    history2.history["val_loss"]
)


# ============================================================
# ACCURACY PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    train_accuracy,
    label="Training Accuracy"
)


plt.plot(
    val_accuracy,
    label="Validation Accuracy"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Hair Condition Gate V2 - Accuracy"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


accuracy_path = (
    RESULTS_DIR
    / "training_accuracy.png"
)


plt.savefig(
    accuracy_path,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# LOSS PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    train_loss,
    label="Training Loss"
)


plt.plot(
    val_loss,
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Hair Condition Gate V2 - Loss"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


loss_path = (
    RESULTS_DIR
    / "training_loss.png"
)


plt.savefig(
    loss_path,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)


print("\nBest Model:")

print(best_model_path)


print("\nFinal Model:")

print(final_model_path)


print("\nClass Names:")

print(class_names_path)


print("\nTraining Accuracy Plot:")

print(accuracy_path)


print("\nTraining Loss Plot:")

print(loss_path)


print("\n" + "=" * 70)