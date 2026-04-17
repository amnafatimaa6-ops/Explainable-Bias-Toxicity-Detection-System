from transformers import pipeline

# Load pretrained bias/toxicity model
classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-hate"
)

# simple toxic keywords
toxic_words = [
    "not good", "inferior", "weak", "bad", "useless",
    "stupid", "lazy", "dumb", "incapable"
]

def analyze_text(text):
    result = classifier(text)[0]

    score = float(result["score"])

    # Risk classification
    if score > 0.7:
        risk_level = "High Risk"
    elif score > 0.4:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    # Find flagged words
    flagged = [w for w in toxic_words if w in text.lower()]

    # AI explanation
    if risk_level == "High Risk":
        explanation = "⚠️ Strong bias detected. This statement contains harmful generalizations."
    elif risk_level == "Moderate Risk":
        explanation = "⚠️ Possible bias or stereotype detected."
    else:
        explanation = "✅ No strong bias detected."

    return {
        "toxicity": round(score, 3),
        "bias": round(score, 3),
        "risk_level": risk_level,
        "flagged_words": flagged,
        "explanation": explanation
    }


def highlight_text(text, words):
    for w in words:
        text = text.replace(w, f"**{w}**")
    return text
