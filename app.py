import streamlit as st
from model import analyze_text
from calibrate import calibrate, risk_level
from evidence import build_evidence
from news import get_news

st.set_page_config(page_title="Trust & Safety AI v5", layout="centered")

st.title("🧠 AI Trust & Safety System v5 (Research Grade)")

# ---------------- INPUT ----------------
text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip():

        result = analyze_text(text)

        bias_risk = calibrate(result["bias_score"])
        tox_risk = calibrate(result["toxicity"])

        evidence = build_evidence(result)

        st.write("## 🧠 Risk Overview")

        st.write("Bias Risk:", risk_level(bias_risk))
        st.write("Toxicity Risk:", risk_level(tox_risk))

        st.write("### 📊 Raw Metrics")
        st.json(result)

        st.write("### 🔍 Evidence-Based Explanation")

        if evidence:
            for e in evidence:
                st.warning(e["reason"])
                if "matched_example" in e:
                    st.info(f"Matched Example: {e['matched_example']}")
        else:
            st.success("No strong harmful patterns detected")

# ---------------- NEWS ----------------
st.subheader("🌍 Live News Intelligence Layer")

news = get_news()

for n in news:
    st.write("## 🧾", n["title"])
    st.write(n["summary"])
    st.write("[Read]", n["link"])
    st.divider()
