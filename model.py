from transformers import pipeline

# Load pretrained model
classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-hate"
)

# patterns for detecting generalizations
bias_patterns = [
    "women are", "men are", "they are", "all people",
    "always", "never", "not good", "inferior", "better than"
]

# possible target groups
target_groups = [
    "women", "men", "immigrants", "muslims",
    "christians", "people", "leaders"
]

def analyze_text(text):
    result = classifier(text)[0]
    score = float(result["score"])

    text_lower = text.lower()

    # detect generalization
    generalization = any(p in text_lower for p in bias_patterns)

    # detect target groups
    targets = [t for t in target_groups if t in text_lower]

    # smarter risk logic
    if generalization and targets:
        risk_level = "High Bias"
    elif score > 0.7:
        risk_level = "High Toxicity"
    elif score > 0.4:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    # explanation logic
    if generalization and targets:
        explanation = f"⚠️ This statement makes a generalized claim about {', '.join(targets)}. Such generalizations can reinforce harmful stereotypes."
    elif score > 0.7:
        explanation = "⚠️ The language appears aggressive or emotionally charged."
    elif score > 0.4:
        explanation = "⚠️ The tone may contain subtle bias or negative framing."
    else:
        explanation = "✅ The statement appears neutral."

    return {
        "toxicity_score": round(score, 3),
        "risk_level": risk_level,
        "targets": targets,
        "flagged_words": targets,  # used for highlighting
        "generalization_detected": generalization,
        "explanation": explanation
    }


def highlight_text(text, words):
    for w in words:
        text = text.replace(w, f"**{w}**")
    return text
