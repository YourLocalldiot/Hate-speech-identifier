"""
Phase 4

TensorFlow Tokenisation
Model Training
Model Saving
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# =============================================================================
# File Paths
# =============================================================================

TRAIN_FILE = Path("datasets/splits/train.csv")
VALID_FILE = Path("datasets/splits/validation.csv")
TEST_FILE = Path("datasets/splits/test.csv")

MODEL_PATH = Path("models/hate_classifier.keras")
TOKENIZER_PATH = Path("models/tokenizer.pkl")

ASSETS = Path("assets")

ASSETS.mkdir(exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)


# =============================================================================
# Hyperparameters
# =============================================================================

VOCAB_SIZE = 20000
MAX_LENGTH = 100

EMBEDDING_DIM = 128

BATCH_SIZE = 32

EPOCHS = 20


# =============================================================================
# Load Dataset
# =============================================================================

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df["text"].astype(str)
X_valid = valid_df["text"].astype(str)
X_test = test_df["text"].astype(str)

y_train = train_df["class"].values
y_valid = valid_df["class"].values
y_test = test_df["class"].values


# =============================================================================
# Tokenizer
# =============================================================================

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)


# =============================================================================
# Convert Text → Sequences
# =============================================================================

X_train = tokenizer.texts_to_sequences(X_train)
X_valid = tokenizer.texts_to_sequences(X_valid)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post",
)

X_valid = pad_sequences(
    X_valid,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post",
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post",
)


# =============================================================================
# Model
# =============================================================================

model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
    ),

    Bidirectional(
        LSTM(64)
    ),

    Dropout(0.5),

    Dense(
        64,
        activation="relu",
    ),

    Dropout(0.3),

    Dense(
        3,
        activation="softmax",
    )
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# =============================================================================
# Callbacks
# =============================================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
    ),

]


# =============================================================================
# Train
# =============================================================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_valid,
        y_valid,
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

)


# =============================================================================
# Evaluate
# =============================================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
)

print("\n")
print("=" * 50)
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Loss     : {loss:.4f}")
print("=" * 50)


# =============================================================================
# Save Model
# =============================================================================

model.save(MODEL_PATH)


# =============================================================================
# Training Curves
# =============================================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training",
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation",
)

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(
    ASSETS / "accuracy.png",
    dpi=300,
)

plt.close()


plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training",
)

plt.plot(
    history.history["val_loss"],
    label="Validation",
)

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    ASSETS / "loss.png",
    dpi=300,
)

plt.close()

print("\nTraining completed successfully.")