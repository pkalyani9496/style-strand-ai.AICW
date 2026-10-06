import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from pathlib import Path
import json
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "datasets" / "hair_presence_prepared" / "train"
VAL_DIR = BASE_DIR / "datasets" / "hair_presence_prepared" / "validation"

MODEL_DIR = BASE_DIR / "models" / "hair_presence"
RESULTS_DIR = BASE_DIR / "results" / "hair_presence"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL PATHS
# ============================================================

BEST_MODEL_PATH = (
    MODEL_DIR / "best_hair_presence_model.keras"
)

MODEL_PATH = (
    MODEL_DIR / "hair_presence_mobilenetv2.keras"
)

CLASS_NAMES_PATH = (
    MODEL_DIR / "class_names.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 70)
print("HAIR / NO-HAIR CLASSIFICATION")
print("=" * 70)

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")


# Save class names

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=4)

print(
    f"\nClass names saved to:\n{CLASS_NAMES_PATH}"
)


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

val_ds = val_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.15
        ),

        layers.RandomZoom(
            0.15
        ),

        layers.RandomContrast(
            0.10
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# BASE MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MOBILENETV2 MODEL")
print("=" * 70)

base_model = MobileNetV2(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)


# Freeze base model for Phase 1

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE PHASE 1
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS - PHASE 1
# ============================================================

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=4,
    restore_best_weights=False,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# PHASE 1 TRAINING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1 - TRANSFER LEARNING")
print("=" * 70)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# PHASE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2 - FINE TUNING")
print("=" * 70)


# Unfreeze the base model

base_model.trainable = True


# Freeze most layers

for layer in base_model.layers[:-100]:
    layer.trainable = False


# Keep BatchNormalization layers frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False


# Recompile with lower learning rate

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# PHASE 2 CALLBACKS
# ============================================================

checkpoint2 = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping2 = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=4,
    restore_best_weights=False,
    verbose=1
)

reduce_lr2 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-8,
    verbose=1
)


# ============================================================
# PHASE 2 TRAINING
# ============================================================

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=[
        checkpoint2,
        early_stopping2,
        reduce_lr2
    ]
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)

print(
    f"\nBest model loaded from:\n{BEST_MODEL_PATH}"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

best_model.save(
    MODEL_PATH
)

print(
    f"\nFinal model saved to:\n{MODEL_PATH}"
)


# ============================================================
# FIND BEST VALIDATION ACCURACY
# ============================================================

val_acc_phase1 = history1.history[
    "val_accuracy"
]

val_acc_phase2 = history2.history[
    "val_accuracy"
]

all_val_acc = (
    val_acc_phase1 +
    val_acc_phase2
)

best_val_accuracy = max(
    all_val_acc
)

best_epoch = (
    all_val_acc.index(
        best_val_accuracy
    ) + 1
)

print("\n" + "=" * 70)
print("BEST VALIDATION RESULT")
print("=" * 70)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Epoch: {best_epoch}"
)


# ============================================================
# TRAINING GRAPHS
# ============================================================

train_accuracy = (
    history1.history["accuracy"] +
    history2.history["accuracy"]
)

validation_accuracy = (
    history1.history["val_accuracy"] +
    history2.history["val_accuracy"]
)

train_loss = (
    history1.history["loss"] +
    history2.history["loss"]
)

validation_loss = (
    history1.history["val_loss"] +
    history2.history["val_loss"]
)

epochs_range = range(
    1,
    len(train_accuracy) + 1
)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs_range,
    train_accuracy,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    validation_accuracy,
    label="Validation Accuracy"
)

plt.axvline(
    len(history1.history["accuracy"]) + 0.5,
    linestyle="--",
    label="Fine-Tuning Start"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Hair / No-Hair Model Accuracy"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

accuracy_path = (
    RESULTS_DIR /
    "training_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=150
)

plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs_range,
    train_loss,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    validation_loss,
    label="Validation Loss"
)

plt.axvline(
    len(history1.history["loss"]) + 0.5,
    linestyle="--",
    label="Fine-Tuning Start"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Hair / No-Hair Model Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

loss_path = (
    RESULTS_DIR /
    "training_loss.png"
)

plt.savefig(
    loss_path,
    dpi=150
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nModel:\n{MODEL_PATH}"
)

print(
    f"\nClass names:\n{CLASS_NAMES_PATH}"
)

print(
    f"\nAccuracy graph:\n{accuracy_path}"
)

print(
    f"\nLoss graph:\n{loss_path}"
)

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(
        f"{i}: {class_name}"
    )

print(
    "\nNext step: "
    "Evaluate the Hair / No-Hair model on the test dataset."
)