import streamlit as st
from model import BiasModel

st.set_page_config(page_title="AI Ethics Monitor", layout="wide")

st.title("🧠 AI Ethics & Bias Intelligence System (Offline Mode)")

st.write("Fully offline AI system for toxicity + bias detection")

# -------------------------
# CACHE MODEL
# -------------------------
@st.cache_resource
def load_model():
    m = BiasModel()
    m.train()
    return m

model = load_model()

# -------------------------
# TEXT ANALYSIS
# -------------------------
st.header("🔍 Analyze Text")

text = st.text_area("Enter text")

if st.button("Analyze") and text:

    result = model.predict(text)

    st.subheader("Prediction")

    st.json(result)

    st.subheader("Key Influencing Words")

    explanation = model.explain(text, "toxicity_label")

    for word, score in explanation:
        st.write(f"{word} → {score:.3f}")

# -------------------------
# SIMULATED “LIVE NEWS”
# -------------------------
st.header("🌍 Simulated News Feed (Offline)")

news_samples = [
    "Government launches new education policy for students",
    "Violence reported in major city causing concern",
    "New AI system improves healthcare and safety",
    "Controversial speech sparks public debate",
]

if st.button("Generate News Analysis"):

    for item in news_samples:

        result = model.predict(item)

        st.markdown("### 📰 " + item)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Toxicity", result["toxicity"])

        with col2:
            st.metric("Bias Signal", result["bias_signal"])

        st.markdown("---")
