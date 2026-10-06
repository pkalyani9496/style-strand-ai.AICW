import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)


# ============================================================
# HAIR SEGMENTATION - U-NET TRAINING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "hair_segmentation_prepared"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "hair_segmentation"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "hair_segmentation"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224

BATCH_SIZE = 8

EPOCHS_PHASE_1 = 15

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GET FILE PAIRS
# ============================================================

def get_pairs(split):

    image_folder = (
        DATASET_ROOT
        / split
        / "images"
    )

    mask_folder = (
        DATASET_ROOT
        / split
        / "masks"
    )

    images = sorted(
        [
            file
            for file in image_folder.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in [".jpg", ".jpeg", ".png"]
        ]
    )

    masks = sorted(
        [
            file
            for file in mask_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".pbm"
            and "(1)" not in file.stem
        ]
    )

    # --------------------------------------------------------
    # Match by normalized filename
    # --------------------------------------------------------

    def normalize(name):

        name = Path(name).stem.lower()

        for suffix in [
            "-org",
            "_org",
            "-gt",
            "_gt",
            "-mask",
            "_mask"
        ]:

            if name.endswith(suffix):

                name = name[
                    :-len(suffix)
                ]

        return name

    mask_dictionary = {}

    for mask in masks:

        key = normalize(mask.name)

        mask_dictionary[key] = mask

    pairs = []

    for image in images:

        key = normalize(image.name)

        if key in mask_dictionary:

            pairs.append(
                (
                    str(image),
                    str(mask_dictionary[key])
                )
            )

    return pairs


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = Image.open(
        path
    ).convert("RGB")

    image = image.resize(
        (IMG_WIDTH, IMG_HEIGHT),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # Normalize 0-255 -> 0-1
    image = image / 255.0

    return image


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(path):

    mask = Image.open(
        path
    ).convert("L")

    # IMPORTANT:
    # nearest-neighbor keeps the binary mask intact
    mask = mask.resize(
        (IMG_WIDTH, IMG_HEIGHT),
        Image.Resampling.NEAREST
    )

    mask = np.asarray(
        mask,
        dtype=np.float32
    )

    # Convert 0/255 -> 0/1
    mask = mask / 255.0

    # Make sure it is binary
    mask = (
        mask > 0.5
    ).astype(np.float32)

    # Add channel dimension
    mask = np.expand_dims(
        mask,
        axis=-1
    )

    return mask


# ============================================================
# DATA GENERATOR
# ============================================================

class SegmentationSequence(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        pairs,
        batch_size=8,
        shuffle=True,
        augment=False
    ):

        self.pairs = pairs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.indices = np.arange(
            len(self.pairs)
        )

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.pairs)
                / self.batch_size
            )
        )

    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_images = []
        batch_masks = []

        for idx in batch_indices:

            image_path, mask_path = (
                self.pairs[idx]
            )

            image = load_image(
                image_path
            )

            mask = load_mask(
                mask_path
            )

            # ------------------------------------------------
            # Data augmentation
            # ------------------------------------------------

            if self.augment:

                # Horizontal flip
                if random.random() < 0.5:

                    image = np.fliplr(
                        image
                    ).copy()

                    mask = np.fliplr(
                        mask
                    ).copy()

                # Small vertical flip
                # Only occasionally
                if random.random() < 0.10:

                    image = np.flipud(
                        image
                    ).copy()

                    mask = np.flipud(
                        mask
                    ).copy()

            batch_images.append(
                image
            )

            batch_masks.append(
                mask
            )

        return (
            np.asarray(
                batch_images,
                dtype=np.float32
            ),
            np.asarray(
                batch_masks,
                dtype=np.float32
            )
        )

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# DICE COEFFICIENT
# ============================================================

def dice_coefficient(
    y_true,
    y_pred
):

    smooth = 1.0

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred,
        tf.float32
    )

    y_true_flat = tf.reshape(
        y_true,
        [-1]
    )

    y_pred_flat = tf.reshape(
        y_pred,
        [-1]
    )

    intersection = tf.reduce_sum(
        y_true_flat
        * y_pred_flat
    )

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        tf.reduce_sum(y_true_flat)
        + tf.reduce_sum(y_pred_flat)
        + smooth
    )

    return dice


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    y_true,
    y_pred
):

    return 1.0 - dice_coefficient(
        y_true,
        y_pred
    )


# ============================================================
# COMBINED LOSS
# ============================================================

def segmentation_loss(
    y_true,
    y_pred
):

    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )

    bce = tf.reduce_mean(
        bce
    )

    d_loss = dice_loss(
        y_true,
        y_pred
    )

    return bce + d_loss


# ============================================================
# IOU METRIC
# ============================================================

def iou_score(
    y_true,
    y_pred
):

    y_pred = tf.cast(
        y_pred > 0.5,
        tf.float32
    )

    y_true = tf.cast(
        y_true > 0.5,
        tf.float32
    )

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=[1, 2, 3]
    )

    union = (
        tf.reduce_sum(
            y_true + y_pred,
            axis=[1, 2, 3]
        )
        - intersection
    )

    iou = (
        intersection + 1.0
    ) / (
        union + 1.0
    )

    return tf.reduce_mean(
        iou
    )


# ============================================================
# CONVOLUTION BLOCK
# ============================================================

def conv_block(
    x,
    filters
):

    x = layers.Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    return x


# ============================================================
# U-NET MODEL
# ============================================================

def build_unet():

    inputs = layers.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        )
    )

    # ========================================================
    # ENCODER
    # ========================================================

    # Block 1
    c1 = conv_block(
        inputs,
        16
    )

    p1 = layers.MaxPooling2D(
        (2, 2)
    )(c1)

    p1 = layers.Dropout(
        0.10
    )(p1)

    # Block 2
    c2 = conv_block(
        p1,
        32
    )

    p2 = layers.MaxPooling2D(
        (2, 2)
    )(c2)

    p2 = layers.Dropout(
        0.10
    )(p2)

    # Block 3
    c3 = conv_block(
        p2,
        64
    )

    p3 = layers.MaxPooling2D(
        (2, 2)
    )(c3)

    p3 = layers.Dropout(
        0.15
    )(p3)

    # Block 4
    c4 = conv_block(
        p3,
        128
    )

    p4 = layers.MaxPooling2D(
        (2, 2)
    )(c4)

    p4 = layers.Dropout(
        0.15
    )(p4)

    # ========================================================
    # BOTTLENECK
    # ========================================================

    c5 = conv_block(
        p4,
        256
    )

    c5 = layers.Dropout(
        0.20
    )(c5)

    # ========================================================
    # DECODER
    # ========================================================

    # Decoder 1
    u6 = layers.UpSampling2D(
        (2, 2)
    )(c5)

    u6 = layers.Concatenate()(
        [u6, c4]
    )

    c6 = conv_block(
        u6,
        128
    )

    # Decoder 2
    u7 = layers.UpSampling2D(
        (2, 2)
    )(c6)

    u7 = layers.Concatenate()(
        [u7, c3]
    )

    c7 = conv_block(
        u7,
        64
    )

    # Decoder 3
    u8 = layers.UpSampling2D(
        (2, 2)
    )(c7)

    u8 = layers.Concatenate()(
        [u8, c2]
    )

    c8 = conv_block(
        u8,
        32
    )

    # Decoder 4
    u9 = layers.UpSampling2D(
        (2, 2)
    )(c8)

    u9 = layers.Concatenate()(
        [u9, c1]
    )

    c9 = conv_block(
        u9,
        16
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c9)

    model = Model(
        inputs,
        outputs,
        name="Hair_Segmentation_UNet"
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HAIR SEGMENTATION - U-NET TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load pairs
    # --------------------------------------------------------

    train_pairs = get_pairs(
        "train"
    )

    val_pairs = get_pairs(
        "val"
    )

    test_pairs = get_pairs(
        "test"
    )

    print("\nDataset:")
    print(
        "Train pairs :",
        len(train_pairs)
    )

    print(
        "Val pairs   :",
        len(val_pairs)
    )

    print(
        "Test pairs  :",
        len(test_pairs)
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if (
        len(train_pairs) == 0
        or len(val_pairs) == 0
        or len(test_pairs) == 0
    ):

        print(
            "\nERROR: Dataset pairs are missing."
        )

        return

    # --------------------------------------------------------
    # Create generators
    # --------------------------------------------------------

    train_generator = SegmentationSequence(
        train_pairs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True
    )

    val_generator = SegmentationSequence(
        val_pairs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding U-Net model...")

    model = build_unet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss=segmentation_loss,
        metrics=[
            dice_coefficient,
            iou_score
        ]
    )

    print("\nModel summary:\n")

    model.summary()

    # ========================================================
    # CALLBACKS
    # ========================================================

    best_model_path = (
        MODEL_ROOT
        / "hair_segmentation_unet_best.keras"
    )

    final_model_path = (
        MODEL_ROOT
        / "hair_segmentation_unet.keras"
    )

    csv_path = (
        RESULTS_ROOT
        / "hair_segmentation_training.csv"
    )

    callbacks = [

        ModelCheckpoint(
            filepath=str(
                best_model_path
            ),
            monitor="val_iou_score",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        EarlyStopping(
            monitor="val_iou_score",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),

        CSVLogger(
            str(csv_path)
        )

    ]

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS_PHASE_1,
        callbacks=callbacks,
        verbose=1
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    model.save(
        final_model_path
    )

    print("\nFinal model saved to:")
    print(
        final_model_path
    )

    print("\nBest model saved to:")
    print(
        best_model_path
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL VALIDATION RESULTS")
    print("=" * 70)

    results = model.evaluate(
        val_generator,
        verbose=1
    )

    for name, value in zip(
        model.metrics_names,
        results
    ):

        print(
            f"{name}: {value:.4f}"
        )

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()