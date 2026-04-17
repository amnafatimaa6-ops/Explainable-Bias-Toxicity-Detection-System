from transformers import pipeline

# Models
toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

sentiment_model = pipeline("sentiment-analysis")

# patterns
bias_patterns = ["women are", "men are", "they are", "all", "always", "never"]
target_groups = ["women", "men", "immigrants", "muslims", "christians", "people"]

violence_keywords = [
    "killed", "murder", "attack", "bomb", "shooting",
    "dead", "war", "violence", "assault"
]

def analyze_text(text):
    text_lower = text.lower()

    # 🔍 MODEL SCORES
    tox = toxicity_model(text)[0]
    sentiment = sentiment_model(text)[0]

    tox_score = float(tox["score"])
    sentiment_label = sentiment["label"]

    # 🔍 DETECT FEATURES
    generalization = any(p in text_lower for p in bias_patterns)
    targets = [t for t in target_groups if t in text_lower]
    violence = any(v in text_lower for v in violence_keywords)

    # 🧠 SMART CLASSIFICATION
    if generalization and targets:
        category = "Bias / Stereotype"
        explanation = f"Statement generalizes about {', '.join(targets)}."
    
    elif violence:
        category = "Violence / Crime"
        explanation = "This text refers to violent or harmful events, not bias."
    
    elif tox_score > 0.7:
        category = "Toxic Language"
        explanation = "Aggressive or offensive tone detected."
    
    elif sentiment_label == "NEGATIVE":
        category = "Negative News"
        explanation = "Negative sentiment but not necessarily biased."
    
    else:
        category = "Neutral"
        explanation = "No strong bias, toxicity, or harmful framing detected."

    return {
        "category": category,
        "toxicity_score": round(tox_score, 3),
        "sentiment": sentiment_label,
        "targets": targets,
        "violence_detected": violence,
        "generalization": generalization,
        "explanation": explanation
    }
