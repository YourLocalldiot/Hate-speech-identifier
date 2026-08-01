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
# Load Resources
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

# Initialise session state so results persist across button clicks
if "result" not in st.session_state:
    st.session_state.result = None

user_input = st.text_area(
    "Enter a sentence",
    height=150,
)

if st.button("Classify"):

    if not user_input.strip():

        st.warning("Please enter some text.")

    else:

        prediction, probabilities = predict(user_input)

        # Store in session state so results remain visible after re-render
        st.session_state.result = {
            "text": user_input,
            "prediction": int(prediction),
            "probabilities": probabilities.tolist(),
        }

# Render results from session state (survives subsequent interactions)
if st.session_state.result is not None:

    r = st.session_state.result
    prediction = r["prediction"]
    probabilities = r["probabilities"]
    label = CLASS_NAMES[prediction]

    if prediction == 0:
        st.success(f"Prediction: **{label}**")

    elif prediction == 1:
        st.warning(f"Prediction: **{label}**")

    else:
        st.error(f"Prediction: **{label}**")

    # ------------------------------------------------------------------
    # Hate Severity Score  (0 = Normal, 100 = Hate Speech)
    # score = P(offensive) * 50 + P(hate) * 100
    # ------------------------------------------------------------------
    score = float(probabilities[1]) * 50 + float(probabilities[2]) * 100
    score = round(min(max(score, 0.0), 100.0), 1)

    st.subheader("Hate Severity Score")

    # Gradient meter rendered with HTML
    bar_html = f"""
    <div style="margin: 0.5rem 0 1.2rem 0;">
        <div style="
            background: linear-gradient(to right, #22c55e, #facc15, #ef4444);
            border-radius: 8px;
            height: 22px;
            width: 100%;
            position: relative;
        ">
            <div style="
                position: absolute;
                left: {score}%;
                top: 50%;
                transform: translate(-50%, -50%);
                width: 14px;
                height: 14px;
                border-radius: 50%;
                background: white;
                border: 2px solid #334155;
                box-shadow: 0 0 4px rgba(0,0,0,0.4);
            "></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#64748b; margin-top:4px;">
            <span>0 &#8212; Normal</span>
            <span>50 &#8212; Offensive</span>
            <span>100 &#8212; Hate Speech</span>
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)
    st.markdown(f"**Score: {score:.1f} / 100**")

    with st.expander("Raw class probabilities"):
        for i in range(3):
            st.progress(float(probabilities[i]))
            st.write(f"{CLASS_NAMES[i]}: {probabilities[i] * 100:.2f}%")