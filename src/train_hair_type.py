from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = Path("datasets/hair_type_prepared/train")
VALIDATION_DIR = Path("datasets/hair_type_prepared/validation")

MODEL_DIR = Path("models/hair_type")
RESULTS_DIR = Path("results/hair_type")

MODEL_PATH = MODEL_DIR / "hair_type_mobilenetv2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

SEED = 42


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("HAIR TYPE MODEL TRAINING")
print("=" * 70)

print("\nTensorFlow version:")
print(tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("\nGPU detected:")
    for gpu in gpus:
        print(gpu)
else:
    print("\nNo GPU detected.")
    print("Training will use CPU.")


# ============================================================
# CHECK DATASET PATHS
# ============================================================

if not TRAIN_DIR.exists():

    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR.resolve()}"
    )


if not VALIDATION_DIR.exists():

    raise FileNotFoundError(
        f"Validation directory not found:\n"
        f"{VALIDATION_DIR.resolve()}"
    )


print("\nTraining directory:")
print(TRAIN_DIR.resolve())

print("\nValidation directory:")
print(VALIDATION_DIR.resolve())


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING VALIDATION DATA")
print("=" * 70)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses detected:")

for index, class_name in enumerate(class_names):

    print(f"{index}: {class_name}")


print(f"\nNumber of classes: {len(class_names)}")


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


print("\nClass names saved to:")
print(CLASS_NAMES_PATH.resolve())


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.10
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        )
    ],
    name="data_augmentation"
)


# ============================================================
# LOAD PRETRAINED MOBILENETV2
# ============================================================

print("\n" + "=" * 70)
print("LOADING MOBILENETV2")
print("=" * 70)

base_model = MobileNetV2(
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)


# ============================================================
# FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = keras.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


x = data_augmentation(inputs)


x = preprocess_input(x)


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


model = keras.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


callbacks = [
    early_stopping,
    reduce_lr,
    checkpoint
]


# ============================================================
# INITIAL TRAINING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1: TRANSFER LEARNING")
print("=" * 70)

print(
    f"\nTraining for maximum {INITIAL_EPOCHS} epochs..."
)

history_initial = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)


# ============================================================
# FINE-TUNING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2: FINE-TUNING")
print("=" * 70)


# Unfreeze the base model
base_model.trainable = True


# Freeze most layers and only fine-tune
# the final part of MobileNetV2

fine_tune_from = 100

for layer in base_model.layers[:fine_tune_from]:

    layer.trainable = False


# Keep BatchNormalization layers frozen
# for more stable training on a small dataset

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# Recompile with a very small learning rate

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print(
    f"\nFine-tuning final MobileNetV2 layers..."
)

history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

model.save(
    MODEL_PATH
)

print("\nModel saved successfully:")
print(MODEL_PATH.resolve())


# ============================================================
# COMBINE TRAINING HISTORY
# ============================================================

initial_accuracy = history_initial.history["accuracy"]
initial_val_accuracy = history_initial.history["val_accuracy"]

fine_accuracy = history_fine.history["accuracy"]
fine_val_accuracy = history_fine.history["val_accuracy"]

initial_loss = history_initial.history["loss"]
initial_val_loss = history_initial.history["val_loss"]

fine_loss = history_fine.history["loss"]
fine_val_loss = history_fine.history["val_loss"]


all_accuracy = (
    initial_accuracy
    + fine_accuracy
)

all_val_accuracy = (
    initial_val_accuracy
    + fine_val_accuracy
)

all_loss = (
    initial_loss
    + fine_loss
)

all_val_loss = (
    initial_val_loss
    + fine_val_loss
)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    all_accuracy,
    label="Training Accuracy"
)

plt.plot(
    all_val_accuracy,
    label="Validation Accuracy"
)

plt.axvline(
    x=len(initial_accuracy) - 1,
    linestyle="--",
    label="Fine-Tuning Start"
)

plt.title(
    "Hair Type Model Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(
    True
)

accuracy_path = (
    RESULTS_DIR
    / "training_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    all_loss,
    label="Training Loss"
)

plt.plot(
    all_val_loss,
    label="Validation Loss"
)

plt.axvline(
    x=len(initial_loss) - 1,
    linestyle="--",
    label="Fine-Tuning Start"
)

plt.title(
    "Hair Type Model Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(
    True
)

loss_path = (
    RESULTS_DIR
    / "training_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH.resolve())

print("\nClass names:")
print(CLASS_NAMES_PATH.resolve())

print("\nTraining graphs:")
print(accuracy_path.resolve())
print(loss_path.resolve())

print("\nNext step:")
print("Run evaluate_hair_type.py")

print("=" * 70)