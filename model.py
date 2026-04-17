from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

tox_model = pipeline("text-classification", model="unitary/toxic-bert")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- ENTITY SIMULATION ----------------
entity_keywords = {
    "person": ["trump", "biden", "elon", "musk"],
    "place": ["uk", "usa", "india", "pakistan", "israel"],
    "event": ["war", "election", "trial", "indictment"]
}

# ---------------- BIAS TYPES ----------------
bias_patterns = {
    "gender": ["men are", "women are"],
    "political": ["government", "president", "party"],
    "racial": ["race", "ethnic", "people are"]
}


def detect_bias_type(text):

    t = text.lower()

    for k, patterns in bias_patterns.items():
        for p in patterns:
            if p in t:
                return k

    return "none"


def detect_entities(text):

    t = text.lower()

    found = []

    for etype, words in entity_keywords.items():
        for w in words:
            if w in t:
                found.append((etype, w))

    return found


def analyze_text(text):

    tox = tox_model(text)[0]

    bias_type = detect_bias_type(text)
    entities = detect_entities(text)

    violence_words = ["kill", "attack", "war", "bomb", "murder"]
    violence_score = sum(w in text.lower() for w in violence_words) / 3

    # ---------------- BIAS SCORE ----------------
    bias_score = 0.4 if bias_type != "none" else 0.1

    if "better than" in text.lower():
        bias_score += 0.3

    return {
        "text": text,
        "toxicity": float(tox["score"]),
        "bias_score": min(bias_score, 1.0),
        "bias_type": bias_type,
        "entities": entities,
        "violence_score": violence_score
    }
