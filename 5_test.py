"""
Phase 5

Model Evaluation
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

from tensorflow.keras.preprocessing.sequence import pad_sequences


# =============================================================================
# File Paths
# =============================================================================

TEST_FILE = Path("datasets/splits/test.csv")

MODEL_PATH = Path("models/hate_classifier.keras")
TOKENIZER_PATH = Path("models/tokenizer.pkl")

ASSETS = Path("assets")

ASSETS.mkdir(exist_ok=True)


# =============================================================================
# Hyperparameters
# =============================================================================

MAX_LENGTH = 100


CLASS_NAMES = [
    "Non-Offensive",
    "Offensive",
    "Hate Speech",
]


# =============================================================================
# Load Model
# =============================================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


# =============================================================================
# Load Test Dataset
# =============================================================================

df = pd.read_csv(TEST_FILE)

texts = df["text"].astype(str)

true_labels = df["class"].values


# =============================================================================
# Tokenise
# =============================================================================

sequences = tokenizer.texts_to_sequences(texts)

X_test = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post",
)


# =============================================================================
# Predict
# =============================================================================

print("Running predictions...")

probabilities = model.predict(X_test)

predictions = probabilities.argmax(axis=1)

confidence = probabilities.max(axis=1)


# =============================================================================
# Metrics
# =============================================================================

accuracy = accuracy_score(
    true_labels,
    predictions,
)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predictions,
    average="macro",
)

print("\n" + "=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Macro F1 : {f1:.4f}")

print("=" * 60)


# =============================================================================
# Classification Report
# =============================================================================

report = classification_report(
    true_labels,
    predictions,
    target_names=CLASS_NAMES,
)

print("\nClassification Report\n")

print(report)

with open(
    ASSETS / "classification_report.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(report)


# =============================================================================
# Confusion Matrix
# =============================================================================

cm = confusion_matrix(
    true_labels,
    predictions,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

plt.figure(figsize=(7, 7))

disp.plot(
    cmap="Blues",
    values_format="d",
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    ASSETS / "confusion_matrix.png",
    dpi=300,
)

plt.close()


# =============================================================================
# Prediction Distribution
# =============================================================================

prediction_counts = (
    pd.Series(predictions)
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(6, 4))

plt.bar(CLASS_NAMES, prediction_counts)

plt.ylabel("Predictions")

plt.title("Prediction Distribution")

plt.tight_layout()

plt.savefig(
    ASSETS / "prediction_distribution.png",
    dpi=300,
)

plt.close()


# =============================================================================
# Misclassified Examples
# =============================================================================

misclassified = df.copy()

misclassified["Actual"] = true_labels

misclassified["Predicted"] = predictions

misclassified["Confidence"] = confidence

misclassified = misclassified[
    misclassified["Actual"] != misclassified["Predicted"]
]

misclassified.to_csv(
    ASSETS / "misclassified_examples.csv",
    index=False,
)

print(f"\nMisclassified examples: {len(misclassified):,}")

print(
    f"Saved to: {ASSETS / 'misclassified_examples.csv'}"
)

print("\nEvaluation complete.")