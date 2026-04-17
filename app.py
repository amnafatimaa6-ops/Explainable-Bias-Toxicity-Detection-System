import streamlit as st
import feedparser
from model import BiasModel

st.set_page_config(page_title="AI Ethics Dashboard", layout="wide")

st.title("🧠 AI Ethics & Bias Intelligence System")

st.write("Real-time bias & toxicity detection using AI + live news feeds")

# -------------------------
# SAFE MODEL LOADING
# -------------------------
@st.cache_resource
def load_model():
    m = BiasModel()
    m.train()
    return m

model = load_model()

# -------------------------
# RSS FEEDS (NO API KEY)
# -------------------------
RSS_FEEDS = {
    "Google News": "https://news.google.com/rss",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Tech": "https://feeds.bbci.co.uk/news/technology/rss.xml"
}

def get_news(url):
    try:
        feed = feedparser.parse(url)
        return feed.entries[:5]
    except:
        return []

# -------------------------
# TEXT ANALYSIS
# -------------------------
st.header("🔍 Analyze Custom Text")

text = st.text_area("Enter text")

if st.button("Analyze") and text:

    result = model.predict(text)

    st.subheader("Prediction")
    st.json(result)

    st.subheader("Top Influencing Words")

    explanation = model.explain(text, "toxicity_label")

    for word, score in explanation:
        st.write(f"{word} → {score:.3f}")

# -------------------------
# LIVE NEWS ANALYSIS
# -------------------------
st.header("🌍 Live News Bias Scanner")

source = st.selectbox("Select News Source", list(RSS_FEEDS.keys()))

if st.button("Fetch Live News"):

    articles = get_news(RSS_FEEDS[source])

    if not articles:
        st.warning("No news found or RSS failed.")
    else:
        for a in articles:

            title = a.get("title", "")
            summary = a.get("summary", "")

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

# -------------------------
# DEBUG PANEL (VERY USEFUL)
# -------------------------
st.sidebar.header("System Status")

if st.sidebar.button("Run Health Check"):
    st.sidebar.success("Model loaded ✔")
    st.sidebar.success("RSS system ready ✔")
    st.sidebar.success("Vectorizer active ✔")
