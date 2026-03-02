import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model('crime_prediction_model_tuned.keras')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load max sequence length
with open('max_sequence_length.pkl', 'rb') as f:
    max_sequence_length = pickle.load(f)

st.title("Crime Prediction App")

user_input = st.text_area("Enter Crime Description")

if st.button("Predict"):
    if user_input.strip() != "":
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_sequence_length)
        prediction = model.predict(padded)
        predicted_label = label_encoder.inverse_transform([prediction.argmax()])[0]
        st.success(f"Predicted Crime Category: {predicted_label}")
    else:
        st.warning("Please enter some text.")