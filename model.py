import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")


class EthicsRadarResearch:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=8000, stop_words="english")
        self.models = {}
        self.sia = SentimentIntensityAnalyzer()

    # -------------------------
    # DATA (SMALL BUT CONTROLLED BASELINE)
    # -------------------------
    def load_data(self):
        df = pd.DataFrame({
            "text": [
                "I love this system",
                "This is horrible and disgusting",
                "Government improves education",
                "Group is dangerous and bad",
                "Women are not good leaders",
                "People are kind and helpful",
                "He is stupid and useless",
                "Education improves society",
                "Violence and war are rising",
                "This is a great achievement",
                "Policy is unfair and biased",
                "Citizens demand justice"
            ],
            "toxicity": [0,1,0,1,1,0,1,0,1,0,1,0],
            "identity_attack": [0,0,0,1,1,0,0,0,0,0,0,0],
            "insult": [0,1,0,1,0,0,1,0,0,0,1,0]
        })

        df["toxicity_label"] = df["toxicity"]
        df["bias_label"] = (
            (df["identity_attack"] == 1) |
            (df["insult"] == 1)
        ).astype(int)

        return df

    # -------------------------
    # TRAINING
    # -------------------------
    def train(self):
        df = self.load_data()

        X = self.vectorizer.fit_transform(df["text"])

        for target in ["toxicity_label", "bias_label"]:
            y = df[target]

            model = LogisticRegression(max_iter=400)
            model.fit(X, y)

            self.models[target] = model

        print("✔ Research model trained")

    # -------------------------
    # SENTIMENT LAYER
    # -------------------------
    def sentiment(self, text):
        s = self.sia.polarity_scores(text)
        return abs(s["compound"])

    # -------------------------
    # FRAMING BIAS DETECTOR (KEY RESEARCH CONTRIBUTION)
    # -------------------------
    def framing_bias(self, text):

        patterns = {
            "generalization": ["all", "always", "never", "every", "most"],
            "negative framing": ["are bad", "are dangerous", "are inferior"],
            "dehumanization": ["animals", "parasites", "threat"]
        }

        score = 0
        t = text.lower()

        for group in patterns.values():
            for p in group:
                if p in t:
                    score += 0.25

        return min(score, 1.0)

    # -------------------------
    # DOMAIN RISK ENGINE
    # -------------------------
    def risk_engine(self, text):

        keywords = {
            "war": 0.4,
            "violence": 0.5,
            "hate": 0.6,
            "kill": 0.7,
            "racist": 0.9,
            "government": 0.1,
            "policy": 0.1,
            "women": 0.15,
            "men": 0.1
        }

        t = text.lower()
        score = sum(v for k, v in keywords.items() if k in t)

        return min(score, 1.0)

    # -------------------------
    # FINAL FUSION MODEL (RESEARCH FORMULA)
    # -------------------------
    def predict(self, text):

        vec = self.vectorizer.transform([text])

        ml_toxic = self.models["toxicity_label"].predict_proba(vec)[0][1]
        ml_bias = self.models["bias_label"].predict_proba(vec)[0][1]

        risk = self.risk_engine(text)
        sentiment = self.sentiment(text)
        framing = self.framing_bias(text)

        # RESEARCH-DEFINED WEIGHTING FUNCTION
        toxicity = (
            0.30 * ml_toxic +
            0.25 * risk +
            0.20 * sentiment +
            0.25 * framing
        )

        bias = (
            0.35 * ml_bias +
            0.30 * framing +
            0.20 * risk +
            0.15 * sentiment
        )

        return {
            "toxicity": round(min(max(toxicity, 0), 1), 3),
            "bias": round(min(max(bias, 0), 1), 3),
            "risk": round(risk, 3),
            "framing_bias": round(framing, 3)
        }

    # -------------------------
    # EXPLAINABILITY LAYER
    # -------------------------
    def explain(self, text):

        vec = self.vectorizer.transform([text])
        features = self.vectorizer.get_feature_names_out()

        model = self.models["toxicity_label"]
        coefs = model.coef_[0]

        scores = []

        for i in vec.nonzero()[1]:
            scores.append((features[i], coefs[i]))

        return sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:10]
