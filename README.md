# Crime Description Classifier

**Live App:** [Add your Streamlit link here]

A Streamlit app that classifies a short, free-text crime description into one of four categories using a lightweight neural text classifier. This was built as a text-classification learning exercise on a small dataset — it is not a production system and is not intended for operational or law-enforcement use.

## What It Does

Given a written description (e.g., "A suspect entered a residence and removed valuables..."), the model predicts which of four categories it most closely matches — Fire Accident, Other Crime, Traffic Fatality, or Violent Crime — and shows a confidence score bucketed into High / Moderate / Requires Further Review.

## Tech Stack, and why each piece is here

- **TensorFlow / Keras** — builds and runs the text classification model.
- **Streamlit** — the app interface, including the confidence-tier display.

## Architecture, and what its size actually means

The model is deliberately small, and it's worth understanding why that matters for how to read its output:

`Embedding (66-word vocabulary, 100-dim) → GlobalAveragePooling1D → Dense(64, relu) → Dropout → Dense(4, softmax)`

- **Embedding layer:** converts each word into a 100-number vector. The vocabulary here is only 66 words — meaning the model only has a learned representation for 66 distinct words total, reflecting the size of the training data it was built on.
- **GlobalAveragePooling1D:** rather than using a more complex layer (like an LSTM) to understand word order, this simply averages the word vectors in a description together into one vector. This is a fast, simple approach, but it means the model treats a description more like a "bag of words" than as an ordered sentence — word order and sentence structure aren't really being used.
- **Dense(64) → Dropout → Dense(4, softmax):** a standard small classifier head that takes that averaged vector and outputs a probability for each of the 4 categories. Dropout is included in the architecture but is currently configured at a rate of 0.0, meaning it isn't actually dropping any connections in the saved model.
- **Input length is capped at 5 tokens** — the model only looks at the first 5 words of whatever you type, which, combined with the 66-word vocabulary, means it's built to handle short, simple phrasing rather than long or varied free-text descriptions.

One more thing worth being upfront about: the app includes a "Contextual Signals Identified" panel that looks like it's explaining the model's reasoning. It isn't — it simply lists the first few unique words from what you typed, not any measure of which words actually influenced the prediction (that would require a technique like attention weights or SHAP values, neither of which is implemented here).

