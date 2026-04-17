import streamlit as st
import feedparser
from model import EthicsRadarV3

st.set_page_config(page_title="Ethics Radar V3", layout="wide")

st.title("🧠 AI Ethics Radar — V3 Research System")

@st.cache_resource
def load():
    m = EthicsRadarV3()
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

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Toxicity", f"{r['toxicity']:.2f}")
    c2.metric("Bias", f"{r['bias']:.2f}")
    c3.metric("Risk", f"{r['risk']:.2f}")
    c4.metric("Framing", f"{r['framing']:.2f}")
    c5.metric("Sentiment", f"{r['sentiment']:.2f}")

    st.success("Analysis complete (calibrated model)")

# -------------------------
# LIVE FEED
# -------------------------
st.header("🌍 Live News Intelligence")

def get_news():
    feed = feedparser.parse("https://news.google.com/rss")
    return [x.title for x in feed.entries[:5]]

if st.button("Run Scan"):

    for item in get_news():

        r = model.predict(item)

        st.markdown("### 🧾 News")
        st.write(item)

        st.write(r)

        st.markdown("---")
