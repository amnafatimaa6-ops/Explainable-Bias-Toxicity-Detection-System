# app.py

import streamlit as st
from model import BiasModel

st.set_page_config(page_title="Bias Detector AI", layout="centered")

st.title("🧠 Bias + Toxicity Detector AI")

st.write("Enter text and the model will analyze toxicity + bias signals.")

# Load model
model = BiasModel()
model.train()

# Input box
text = st.text_area("Enter text here")

if st.button("Analyze"):

    result = model.predict(text)

    st.subheader("Prediction Results")

    st.write(result)

    st.subheader("Top Influencing Words")

    explanation = model.explain(text, "toxicity_label")

    for word, score in explanation:
        st.write(f"{word} → {score:.3f}")
