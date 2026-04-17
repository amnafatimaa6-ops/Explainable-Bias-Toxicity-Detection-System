import streamlit as st
import requests
from model import BiasModel

# ----------------------------
# INIT MODEL
# ----------------------------
model = BiasModel()
model.train()

# ----------------------------
# NEWS API CONFIG
# ----------------------------
NEWS_API_KEY = "YOUR_API_KEY_HERE"

def get_news(query="technology"):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=5&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    return response.json().get("articles", [])


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="AI Ethics Monitor", layout="wide")

st.title("🧠 AI Ethics & Bias Intelligence System")

st.write("""
A real-time AI system that detects toxicity, bias signals,
and analyzes live news articles for ethical risks.
""")

# ----------------------------
# USER INPUT SECTION
# ----------------------------
st.header("🔍 Analyze Custom Text")

text = st.text_area("Enter text")

if st.button("Analyze Text"):

    result = model.predict(text)

    st.subheader("Prediction")
    st.json(result)

    st.subheader("Key Influencing Words")

    explanation = model.explain(text, "toxicity_label")

    for word, score in explanation:
        st.write(f"**{word}** → {score:.3f}")


# ----------------------------
# LIVE NEWS ANALYSIS SECTION
# ----------------------------
st.header("🌍 Live News Bias Scanner")

query = st.text_input("Search news topic (e.g. politics, AI, war, education)", "technology")

if st.button("Fetch & Analyze News"):

    articles = get_news(query)

    if not articles:
        st.warning("No articles found or API issue.")
    else:
        for article in articles:

            content = article.get("title", "") + " " + str(article.get("description", ""))

            if not content.strip():
                continue

            result = model.predict(content)

            st.markdown("### 📰 " + article["title"])
            st.write(article["description"])

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Toxicity:**", result["toxicity"])

            with col2:
                st.write("**Bias Signal:**", result["bias_signal"])

            st.markdown("---")
