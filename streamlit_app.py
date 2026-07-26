"""
Phase 6

Streamlit Web Application
"""

from pathlib import Path
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences


# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Hate Speech Classifier",
    page_icon="💬",
    layout="centered",
)

MODEL_PATH = Path("models/hate_classifier.keras")
TOKENIZER_PATH = Path("models/tokenizer.pkl")

MAX_LENGTH = 100

CLASS_NAMES = [
    "Non-Offensive",
    "Offensive",
    "Hate Speech",
]


# =============================================================================
# Load Model
# =============================================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()
tokenizer = load_tokenizer()


# =============================================================================
# Prediction
# =============================================================================

def predict(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(
        padded,
        verbose=0,
    )[0]

    prediction = np.argmax(probabilities)

    return prediction, probabilities


# =============================================================================
# Interface
# =============================================================================

st.title("Hate Speech Detection")

st.write(
    """
    Detect whether a sentence is:

    - Non-Offensive
    - Offensive
    - Hate Speech
    """
)

user_input = st.text_area(
    "Enter a sentence",
    height=150,
)

if st.button("Classify"):

    if not user_input.strip():

        st.warning("Please enter some text.")

    else:

        prediction, probabilities = predict(user_input)

        label = CLASS_NAMES[prediction]

        if prediction == 0:
            st.success(f"Prediction: **{label}**")

        elif prediction == 1:
            st.warning(f"Prediction: **{label}**")

        else:
            st.error(f"Prediction: **{label}**")

        st.subheader("Confidence")

        for i in range(3):

            st.progress(float(probabilities[i]))

            st.write(
                f"{CLASS_NAMES[i]}: {probabilities[i]*100:.2f}%"
            )