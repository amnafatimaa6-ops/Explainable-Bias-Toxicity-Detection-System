import streamlit as st
import requests
import feedparser
from model import BiasModel

st.set_page_config(page_title="AI Ethics Monitor", layout="wide")

st.title("🧠 AI Ethics & Bias Intelligence System (Live Offline Mode)")

st.write("Analyzing real-world news using AI (no API keys needed).")

# ---------------------------
# LOAD MODEL (CACHED)
# ---------------------------
@st.cache_resource
def load_model():
    m = BiasModel()
    m.train()
    return m

model = load_model()

# ---------------------------
# LIVE RSS SOURCES (NO API KEY)
# ---------------------------
RSS_FEEDS = {
    "Google News": "https://news.google.com/rss",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Tech": "https://feeds.bbci.co.uk/news/technology/rss.xml"
}

def get_news(feed_url):
    feed = feedparser.parse(feed_url)
    return feed.entries[:5]


# ---------------------------
# TEXT ANALYSIS
# ---------------------------
st.header("🔍 Analyze Your Text")

text = st.text_area("Enter text")

if st.button("Analyze Text") and text:

    result = model.predict(text)

    st.subheader("Prediction")
    st.json(result)

    st.subheader("Key Influencing Words")

    explanation = model.explain(text, "toxicity_label")

    for word, score in explanation:
        st.write(f"{word} → {score:.3f}")


# ---------------------------
# LIVE NEWS ANALYSIS
# ---------------------------
st.header("🌍 Live News Bias Scanner (No API Required)")

source = st.selectbox("Select News Source", list(RSS_FEEDS.keys()))

if st.button("Fetch Live News"):

    articles = get_news(RSS_FEEDS[source])

    for article in articles:

        title = article.get("title", "")
        summary = article.get("summary", "")

        content = title + " " + summary

        if not content.strip():
            continue

        result = model.predict(content)

        st.markdown("### 📰 " + title)

        st.write(summary)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Toxicity", result["toxicity"])

        with col2:
            st.metric("Bias Signal", result["bias_signal"])

        st.markdown("---")
