import streamlit as st
import feedparser
from model import EthicsRadarResearch

st.set_page_config(page_title="AI Ethics Radar Research", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>🧠 AI Ethics Radar — Research Edition</h1>
<p style='text-align:center;'>Hybrid ML + Linguistic Bias Detection System</p>
""", unsafe_allow_html=True)

# -------------------------
# MODEL
# -------------------------
@st.cache_resource
def load():
    m = EthicsRadarResearch()
    m.train()
    return m

model = load()

# -------------------------
# INPUT
# -------------------------
st.header("🔍 Text Analysis")
text = st.text_area("Enter text")

if st.button("Analyze") and text:

    r = model.predict(text)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Toxicity", f"{r['toxicity']:.2f}")
    c2.metric("Bias", f"{r['bias']:.2f}")
    c3.metric("Risk", f"{r['risk']:.2f}")
    c4.metric("Framing", f"{r['framing_bias']:.2f}")

    st.subheader("Explainability Layer")

    for word, score in model.explain(text):
        color = "red" if score < 0 else "green"
        st.markdown(f"<span style='color:{color}; font-size:18px'>{word}</span>", unsafe_allow_html=True)

# -------------------------
# LIVE NEWS ANALYSIS
# -------------------------
st.header("🌍 Live Media Bias Scan")

def get_news():
    feed = feedparser.parse("https://news.google.com/rss")
    return [x.title for x in feed.entries[:6]]

if st.button("Run Live Scan"):

    for item in get_news():

        r = model.predict(item)

        st.markdown("### 🧾 News")
        st.write(item)

        st.write(r)

        st.markdown("---")
