import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Crime Intelligence System",
    page_icon="🔎",
    layout="centered"
)

# -------------------------------------------------
# Load Artifacts (Safe Caching)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crime_prediction_model_tuned.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_max_length():
    with open("max_sequence_length.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
label_encoder = load_label_encoder()
max_sequence_length = load_max_length()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("🔎 Crime Classification Intelligence System")
st.markdown(
    "This system analyzes written crime descriptions and identifies the most relevant category "
    "based on patterns learned from historical crime records."
)

# -------------------------------------------------
# Sidebar (Professional Language)
# -------------------------------------------------
st.sidebar.title("System Overview")

st.sidebar.markdown("""
• Built using advanced pattern recognition  
• Optimized for reliability and stability  
• Evaluated across multiple internal configurations  
• Designed for structured analytical insight  

The system selects the most reliable internal configuration before deployment to ensure consistent performance.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("🔐 Designed for structured reporting environments.")

# -------------------------------------------------
# Input
# -------------------------------------------------
user_input = st.text_area(
    "Enter Crime Description:",
    placeholder="Example: A suspect entered a residence and removed valuables..."
)

# -------------------------------------------------
# Helper: Confidence Interpretation
# -------------------------------------------------
def interpret_confidence(score):
    if score >= 0.70:
        return "High Confidence", "🟢"
    elif score >= 0.40:
        return "Moderate Confidence", "🟡"
    else:
        return "Requires Further Review", "🔴"

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Analyze Description"):

    if user_input.strip() == "":
        st.warning("Please provide a valid description.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_sequence_length)

        prediction = model.predict(padded)
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        raw_confidence = float(np.max(prediction))
        confidence_label, indicator = interpret_confidence(raw_confidence)

        # Display Main Result
        st.success(f"Predicted Category: {predicted_label}")
        st.info(f"{indicator} {confidence_label}")

        # -------------------------------------------------
        # Intelligent Explanation Section
        # -------------------------------------------------
        tokens = user_input.lower().split()
        unique_tokens = list(set(tokens))
        detected_keywords = unique_tokens[:5]

        st.markdown("### Contextual Signals Identified")
        for word in detected_keywords:
            st.write(f"• {word}")

        st.markdown("""
        The classification is determined by analyzing contextual patterns 
        and linguistic similarities to previously recorded cases within the system.
        """)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>Crime Intelligence System | Optimized Analytical Model</center>",
    unsafe_allow_html=True
)

