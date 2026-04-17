import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")


class EthicsRadarStable:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1,2),
            stop_words="english"
        )

        self.models = {}
        self.sia = SentimentIntensityAnalyzer()

    # -------------------------
    # SAFE DATA LOADING (FIXED)
    # -------------------------
    def load_data(self):

        dataset = load_dataset("civil_comments", split="train[:5%]")
        df = pd.DataFrame(dataset)

        df = df[["text", "toxicity", "identity_attack", "insult"]].dropna()

        df["toxicity_label"] = (df["toxicity"] > 0.5).astype(int)

        df["bias_label"] = (
            (df["identity_attack"] > 0.5) |
            (df["insult"] > 0.7)
        ).astype(int)

        return df[["text", "toxicity_label", "bias_label"]]

    # -------------------------
    # TRAIN (NO CRASH VERSION)
    # -------------------------
    def train(self):

        df = self.load_data()

        X = self.vectorizer.fit_transform(df["text"])

        for target in ["toxicity_label", "bias_label"]:

            y = df[target]

            model = LogisticRegression(
                max_iter=300,
                class_weight="balanced"
            )

            model.fit(X, y)

            self.models[target] = model

        print("✔ Stable model trained")

    # -------------------------
    # SENTIMENT
    # -------------------------
    def sentiment(self, text):
        return abs(self.sia.polarity_scores(text)["compound"])

    # -------------------------
    # FRAMING BIAS
    # -------------------------
    def framing_bias(self, text):

        patterns = [
            "are bad", "are evil", "are dangerous",
            "all", "always", "never"
        ]

        t = text.lower()

        score = sum(0.2 for p in patterns if p in t)

        return min(score, 1.0)

    # -------------------------
    # RISK ENGINE
    # -------------------------
    def risk_engine(self, text):

        keywords = {
            "war": 0.4,
            "hate": 0.6,
            "violence": 0.5,
            "kill": 0.7,
            "racist": 0.9,
            "crime": 0.3,
            "government": 0.1,
            "women": 0.1
        }

        t = text.lower()

        score = sum(v for k,v in keywords.items() if k in t)

        return min(score, 1.0)

    # -------------------------
    # NORMALIZE (IMPORTANT)
    # -------------------------
    def norm(self, x):
        return float(np.clip(x, 0.05, 0.95))

    # -------------------------
    # PREDICT (SAFE VERSION)
    # -------------------------
    def predict(self, text):

        X = self.vectorizer.transform([text])

        ml_toxic = self.models["toxicity_label"].predict_proba(X)[0][1]
        ml_bias = self.models["bias_label"].predict_proba(X)[0][1]

        risk = self.risk_engine(text)
        sentiment = self.sentiment(text)
        framing = self.framing_bias(text)

        toxicity = (
            0.45 * ml_toxic +
            0.25 * risk +
            0.20 * framing +
            0.10 * sentiment
        )

        bias = (
            0.45 * ml_bias +
            0.25 * framing +
            0.20 * risk +
            0.10 * sentiment
        )

        return {
            "toxicity": self.norm(toxicity),
            "bias": self.norm(bias),
            "risk": self.norm(risk),
            "framing": self.norm(framing),
            "sentiment": self.norm(sentiment)
        }
