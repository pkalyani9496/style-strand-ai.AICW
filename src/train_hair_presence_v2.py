import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# 1. PATHS
# ============================================================

TRAIN_DIR = "datasets/hair_presence_prepared_v2/train"
VAL_DIR = "datasets/hair_presence_prepared_v2/validation"

MODEL_DIR = "models/hair_presence_v2"
RESULTS_DIR = "results/hair_presence_v2"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_hair_presence_v2.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hair_presence_mobilenetv2_v2.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.json"
)


# ============================================================
# 2. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# 4. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 5. CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\nClass names:")

for i, name in enumerate(class_names):

    print(f"{i}: {name}")


# Save class names

with open(CLASS_NAMES_PATH, "w") as f:

    json.dump(
        class_names,
        f,
        indent=4
    )


# ============================================================
# 6. PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

val_ds = val_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# 7. DATA AUGMENTATION
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
            0.15
        )
    ],
    name="data_augmentation"
)


# ============================================================
# 8. LOAD MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# ============================================================
# 9. FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# 10. BUILD MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)


x = data_augmentation(inputs)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dropout(
    0.3
)(x)


outputs = layers.Dense(
    2,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 12. CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    mode="max",
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
# 13. PHASE 1 TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("PHASE 1 - TRAINING CLASSIFICATION HEAD")
print("=" * 60)


history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# 14. PHASE 2 - FINE TUNING
# ============================================================

print("\n")
print("=" * 60)
print("PHASE 2 - FINE TUNING")
print("=" * 60)


base_model.trainable = True


# Freeze most layers.
# Only the last 100 layers will be trainable.

for layer in base_model.layers[:-100]:

    layer.trainable = False


# Keep BatchNormalization layers frozen.

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# Recompile with lower learning rate.

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# 15. LOAD BEST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)


best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

best_model.save(
    FINAL_MODEL_PATH
)


print("\nBest model saved to:")

print(
    os.path.abspath(
        BEST_MODEL_PATH
    )
)


print("\nFinal model saved to:")

print(
    os.path.abspath(
        FINAL_MODEL_PATH
    )
)


# ============================================================
# 17. FIND BEST VALIDATION ACCURACY
# ============================================================

phase1_best = max(
    history1.history["val_accuracy"]
)

phase2_best = max(
    history2.history["val_accuracy"]
)

overall_best = max(
    phase1_best,
    phase2_best
)


print("\n")
print("=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

print(
    f"Phase 1 Best Validation Accuracy: "
    f"{phase1_best * 100:.2f}%"
)

print(
    f"Phase 2 Best Validation Accuracy: "
    f"{phase2_best * 100:.2f}%"
)

print(
    f"Overall Best Validation Accuracy: "
    f"{overall_best * 100:.2f}%"
)


# ============================================================
# 18. SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    RESULTS_DIR,
    "training_history.json"
)


combined_history = {

    "phase1": history1.history,

    "phase2": history2.history

}


with open(
    history_path,
    "w"
) as f:

    json.dump(
        combined_history,
        f,
        indent=4
    )


# ============================================================
# 19. PLOT ACCURACY
# ============================================================

import matplotlib.pyplot as plt


phase1_acc = history1.history["accuracy"]

phase1_val_acc = history1.history["val_accuracy"]

phase2_acc = history2.history["accuracy"]

phase2_val_acc = history2.history["val_accuracy"]


train_acc = (
    phase1_acc +
    phase2_acc
)

val_acc = (
    phase1_val_acc +
    phase2_val_acc
)


plt.figure(figsize=(10, 6))

plt.plot(
    train_acc,
    label="Training Accuracy"
)

plt.plot(
    val_acc,
    label="Validation Accuracy"
)

plt.title(
    "Hair Presence V2 - Training Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()


accuracy_path = os.path.join(
    RESULTS_DIR,
    "training_accuracy.png"
)

plt.savefig(
    accuracy_path
)

plt.close()


# ============================================================
# 20. PLOT LOSS
# ============================================================

phase1_loss = history1.history["loss"]

phase1_val_loss = history1.history["val_loss"]

phase2_loss = history2.history["loss"]

phase2_val_loss = history2.history["val_loss"]


train_loss = (
    phase1_loss +
    phase2_loss
)

val_loss = (
    phase1_val_loss +
    phase2_val_loss
)


plt.figure(figsize=(10, 6))

plt.plot(
    train_loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
    label="Validation Loss"
)

plt.title(
    "Hair Presence V2 - Training Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()


loss_path = os.path.join(
    RESULTS_DIR,
    "training_loss.png"
)

plt.savefig(
    loss_path
)

plt.close()


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("HAIR PRESENCE V2 TRAINING COMPLETED")
print("=" * 60)

print("\nClass names saved to:")
print(
    os.path.abspath(
        CLASS_NAMES_PATH
    )
)

print("\nAccuracy graph saved to:")
print(
    os.path.abspath(
        accuracy_path
    )
)

print("\nLoss graph saved to:")
print(
    os.path.abspath(
        loss_path
    )
)

print("\nDone!")