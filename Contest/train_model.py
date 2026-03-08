import pandas as pd
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.layers import (
    Input,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    Lambda
)

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# ==========================================
# 0. GPU CONFIG
# ==========================================

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPU detected:", gpus)

    # allow dynamic memory allocation
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

else:
    print("No GPU detected")


# ==========================================
# 1. Load CSV
# ==========================================

SURVEY_CSV = "data/data_from_questionaire.csv"
IG_CSV = "data/data_from_intragram.csv"

SURVEY_IMG_DIR = "data/questionnaire_images/"
IG_IMG_DIR = "data/instagram_images/"

df_survey = pd.read_csv(SURVEY_CSV)
df_ig = pd.read_csv(IG_CSV)

df_survey["Image 1"] = SURVEY_IMG_DIR + df_survey["Image 1"]
df_survey["Image 2"] = SURVEY_IMG_DIR + df_survey["Image 2"]

df_ig["Image 1"] = IG_IMG_DIR + df_ig["Menu"] + "/" + df_ig["Image 1"]
df_ig["Image 2"] = IG_IMG_DIR + df_ig["Menu"] + "/" + df_ig["Image 2"]

df_all = pd.concat(
    [
        df_survey[["Image 1", "Image 2", "Menu", "Winner"]],
        df_ig[["Image 1", "Image 2", "Menu", "Winner"]],
    ],
    ignore_index=True,
)

df_all["label"] = df_all["Winner"].apply(lambda x: 0 if x == 1 else 1)

train_df, val_df = train_test_split(
    df_all,
    test_size=0.2,
    random_state=42,
    stratify=df_all["Menu"],
)

print("Train:", len(train_df))
print("Validation:", len(val_df))


# ==========================================
# 2. Dataset Pipeline
# ==========================================

IMG_SIZE = 224


def process_image(path):

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)

    img = preprocess_input(img)

    return img


def load_pair(path1, path2, label):

    img1 = process_image(path1)
    img2 = process_image(path2)

    return {"image_1": img1, "image_2": img2}, label


def create_dataset(dataframe, batch_size=32, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            dataframe["Image 1"].values,
            dataframe["Image 2"].values,
            dataframe["label"].values,
        )
    )

    dataset = dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


BATCH_SIZE = 32

train_dataset = create_dataset(train_df, BATCH_SIZE, True)
val_dataset = create_dataset(val_df, BATCH_SIZE, False)


# ==========================================
# 3. Siamese Model
# ==========================================

def build_siamese_model(input_shape=(224, 224, 3)):

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    base_model.trainable = False

    input_layer = Input(shape=input_shape)

    x = base_model(input_layer, training=False)
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    embedding = Dense(128)(x)

    feature_extractor = Model(input_layer, embedding)

    input_1 = Input(shape=input_shape, name="image_1")
    input_2 = Input(shape=input_shape, name="image_2")

    feat1 = feature_extractor(input_1)
    feat2 = feature_extractor(input_2)

    distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))(
        [feat1, feat2]
    )

    x = Dense(64, activation="relu")(distance)
    x = Dropout(0.2)(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[input_1, input_2], outputs=output)

    return model


# ==========================================
# 4. Compile (FORCE GPU)
# ==========================================

with tf.device('/GPU:0'):

    model = build_siamese_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

model.summary()


# ==========================================
# 5. Callbacks
# ==========================================

callbacks = [

    ModelCheckpoint(
        "best_siamese_model.h5",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
]


# ==========================================
# 6. Train
# ==========================================

print("Start Training...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=callbacks,
)

print("Training Finished")