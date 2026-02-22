import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input

# ======================
# PATHS
# ======================
OLD_MODEL_PATH = r"C:\Users\Aarav Gupta\OneDrive\Desktop\segregation and stuff\model.h5"
DATASET_PATH = r"C:\Users\Aarav Gupta\OneDrive\Desktop\waste_dataset"

IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 20

# ======================
# LOAD DATASET
# ======================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{DATASET_PATH}/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{DATASET_PATH}/val",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Original classes:", class_names)

# ======================
# CONVERT LABELS → BINARY
# ======================
recyclable_classes = ["cardboard", "glass", "metal", "paper", "plastic"]
recyclable_indices = [class_names.index(c) for c in recyclable_classes]

def to_binary(images, labels):
    binary_labels = tf.where(
        tf.reduce_any(
            tf.equal(tf.expand_dims(labels, -1), recyclable_indices),
            axis=1
        ),
        0,  # Recyclable
        1   # Non-Recyclable
    )
    return images, binary_labels

train_ds = train_ds.map(to_binary).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(to_binary).prefetch(tf.data.AUTOTUNE)

# ======================
# LOAD OLD MODEL
# ======================
old_model = load_model(OLD_MODEL_PATH)

# Freeze all old layers
old_model.trainable = False

# ======================
# WRAP OLD MODEL CORRECTLY
# ======================
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
features = old_model(inputs)          # CALL the model
old_model.summary()
outputs = Dense(2, activation="softmax")(features)

binary_model = Model(inputs, outputs)

# ======================
# COMPILE
# ======================
binary_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

binary_model.summary()

# ======================
# TRAIN
# ======================
binary_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ======================
# SAVE
# ======================
binary_model.save("model_binary.h5")

print("✅ Binary fine-tuning completed")
print("✅ Saved as model_binary.h5")
