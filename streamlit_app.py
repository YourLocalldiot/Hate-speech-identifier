"""
Phase 6

Streamlit Web Application
"""

import os
from pathlib import Path
import pickle

from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences


# =============================================================================
# Configuration
# =============================================================================

load_dotenv(Path(__file__).parent / ".env")

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

# Configure Gemini
_gemini_key = os.getenv("GEMINI_API_KEY", "")
if _gemini_key:
    genai.configure(api_key=_gemini_key)


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


@st.cache_resource
def load_gemini():
    if not _gemini_key:
        return None
    return genai.GenerativeModel("gemini-2.0-flash")


model = load_model()
tokenizer = load_tokenizer()
gemini = load_gemini()


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
# Gemini Explanation
# =============================================================================

def explain_with_gemini(text: str, label: str, score: float, probabilities: list) -> str:
    """Ask Gemini to explain why the classifier assigned this label."""

    if gemini is None:
        return "⚠️ Gemini API key not configured."

    prob_lines = "\n".join(
        f"  - {CLASS_NAMES[i]}: {probabilities[i] * 100:.1f}%"
        for i in range(3)
    )

    prompt = f"""You are a content moderation expert helping users understand an AI hate-speech classifier.

A Bidirectional LSTM model (trained on the Davidson et al. Twitter hate-speech dataset) has analysed the following text:

Text: "{text}"

Classification result:
  - Label: {label}
  - Hate Severity Score: {score:.1f} / 100  (0 = Normal, 50 = Offensive, 100 = Hate Speech)
  - Raw class probabilities:
{prob_lines}

Please explain IN 3–5 SHORT BULLET POINTS why the model likely assigned this result.
Cover:
1. Which specific words or phrases likely triggered the classification
2. Whether the result seems accurate or potentially wrong (with reasoning)
3. Any nuance or context that might affect the label

Be concise, plain-spoken, and educational. Do NOT repeat the score or raw probabilities in your answer."""

    try:
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"


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

if "explanation" not in st.session_state:
    st.session_state.explanation = None

user_input = st.text_area(
    "Enter a sentence",
    height=150,
)

if st.button("Classify"):

    if not user_input.strip():

        st.warning("Please enter some text.")

    else:

        prediction, probabilities = predict(user_input)

        # Clear old explanation when a new classification is run
        st.session_state.explanation = None

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

    # ------------------------------------------------------------------
    # Gemini Explanation
    # ------------------------------------------------------------------
    st.divider()

    if st.button("✨ Explain with Gemini"):
        with st.spinner("Asking Gemini..."):
            st.session_state.explanation = explain_with_gemini(
                text=r["text"],
                label=label,
                score=score,
                probabilities=probabilities,
            )

    if st.session_state.explanation:
        st.subheader("Gemini Explanation")
        st.markdown(st.session_state.explanation)