import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "datasets", "hair_disease", "train")
VAL_DIR = os.path.join(BASE_DIR, "datasets", "hair_disease", "val")
TEST_DIR = os.path.join(BASE_DIR, "datasets", "hair_disease", "test")

MODEL_DIR = os.path.join(BASE_DIR, "models", "hair_disease")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "hair_disease")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# 3. LOAD DATASETS
# ============================================================

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

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 4. CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\nClasses:")
for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")

print(f"\nNumber of classes: {len(class_names)}")

# Save class names
class_names_path = os.path.join(MODEL_DIR, "class_names.json")

with open(class_names_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

print(f"\nClass names saved to: {class_names_path}")


# ============================================================
# 5. PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
], name="data_augmentation")


# ============================================================
# 7. MOBILE NET V2 BASE MODEL
# ============================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model for Phase 1
base_model.trainable = False


# ============================================================
# 8. BUILD MODEL
# ============================================================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)


# ============================================================
# 9. COMPILE - PHASE 1
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()


# ============================================================
# 10. CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_hair_disease_model.keras"
)

callbacks = [
    ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]


# ============================================================
# 11. PHASE 1 TRAINING
# ============================================================

print("\n" + "=" * 60)
print("PHASE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 60)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)


# ============================================================
# 12. PHASE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2: FINE-TUNING MOBILE NET V2")
print("=" * 60)

base_model.trainable = True

# Freeze earlier layers
for layer in base_model.layers[:-100]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)


# ============================================================
# 13. LOAD BEST MODEL
# ============================================================

print("\nLoading best checkpoint...")

best_model = tf.keras.models.load_model(best_model_path)


# ============================================================
# 14. TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)

test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# 15. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "hair_disease_mobilenetv2.keras"
)

best_model.save(final_model_path)

print(f"\nFinal model saved to:")
print(final_model_path)


# ============================================================
# 16. SAVE TRAINING HISTORY
# ============================================================

import matplotlib.pyplot as plt

acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]

loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]

epochs_range = range(1, len(acc) + 1)


# Accuracy plot
plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    acc,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_acc,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Hair Disease Model - Training and Validation Accuracy")
plt.legend()
plt.grid(True)

accuracy_plot_path = os.path.join(
    RESULTS_DIR,
    "training_accuracy.png"
)

plt.savefig(
    accuracy_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Loss plot
plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    loss,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_loss,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Hair Disease Model - Training and Validation Loss")
plt.legend()
plt.grid(True)

loss_plot_path = os.path.join(
    RESULTS_DIR,
    "training_loss.png"
)

plt.savefig(
    loss_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nTraining plots saved:")
print(accuracy_plot_path)
print(loss_plot_path)

print("\n" + "=" * 60)
print("HAIR DISEASE TRAINING COMPLETED")
print("=" * 60)