import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Crime Insight System",
    page_icon="🔍",
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
st.sidebar.title("About This System")

st.sidebar.markdown("""
• Built using advanced pattern recognition  
• Optimized for higher reliability and accuracy  
• Evaluated across multiple training configurations  
• Designed for consistent performance  

This system improves prediction quality by testing multiple internal configurations 
and selecting the most reliable one before deployment.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("🔐 Designed for analytical insight and structured crime reporting.")

# -------------------------------------------------
# Input Section
# -------------------------------------------------
user_input = st.text_area(
    "Enter Crime Description:",
    placeholder="Example: A suspect entered a residence and removed valuables without permission..."
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Analyze Description"):

    if user_input.strip() == "":
        st.warning("Please provide a valid description to analyze.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_sequence_length)

        prediction = model.predict(padded)
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        confidence = float(np.max(prediction)) * 100

        st.success(f"Predicted Category: {predicted_label}")
        st.info(f"Confidence Level: {confidence:.2f}%")

        # Probability Visualization
        probs = prediction.flatten()
        labels = label_encoder.classes_

        prob_df = pd.DataFrame({
            "Category": labels,
            "Likelihood": probs
        }).sort_values(by="Likelihood", ascending=True)

        st.markdown("### Likelihood Breakdown")
        fig, ax = plt.subplots()
        ax.barh(prob_df["Category"], prob_df["Likelihood"])
        ax.set_xlabel("Relative Likelihood")
        st.pyplot(fig)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>Crime Insight System | Optimized Prediction Model</center>",
    unsafe_allow_html=True
)
