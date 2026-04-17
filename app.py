from transformers import pipeline

# Toxicity model
toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

# Sentiment model
sentiment_model = pipeline("sentiment-analysis")

bias_patterns = ["women are", "men are", "they are", "all", "always", "never"]
target_groups = ["women", "men", "immigrants", "muslims", "christians", "people"]
violence_keywords = ["killed", "murder", "attack", "bomb", "shooting", "war", "violence"]


def analyze_text(text):
    text_lower = text.lower()

    tox = toxicity_model(text)[0]
    sentiment = sentiment_model(text)[0]

    tox_score = float(tox["score"])
    sentiment_label = sentiment["label"]

    generalization = any(p in text_lower for p in bias_patterns)
    targets = [t for t in target_groups if t in text_lower]
    violence = any(v in text_lower for v in violence_keywords)

    # CATEGORY LOGIC
    if generalization and targets:
        category = "Bias / Stereotype"
        explanation = f"This statement generalizes about {', '.join(targets)}."

    elif violence:
        category = "Violence / Crime Context"
        explanation = "This text refers to violence or crime events, not bias."

    elif tox_score > 0.7:
        category = "Toxic Language"
        explanation = "Highly aggressive or offensive language detected."

    elif sentiment_label == "NEGATIVE":
        category = "Negative News"
        explanation = "Negative sentiment but not necessarily biased."

    else:
        category = "Neutral"
        explanation = "No significant bias or toxicity detected."

    return {
        "category": category,
        "toxicity_score": round(tox_score, 3),
        "sentiment": sentiment_label,
        "targets": targets,
        "generalization": generalization,
        "explanation": explanation
    }


# SAFE highlight function (NO IMPORT ERRORS EVER)
def highlight_text(text, words):
    if not words:
        return text

    for w in words:
        text = text.replace(w, f"**{w}**")
    return text
