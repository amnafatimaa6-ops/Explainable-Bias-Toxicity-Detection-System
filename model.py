import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")


class EthicsRadarSafe:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1,2),
            stop_words="english"
        )

        self.models = {}
        self.sia = SentimentIntensityAnalyzer()

    # -------------------------
    # SMALL INTERNAL TRAINING DATA (NO INTERNET)
    # -------------------------
    def load_data(self):

        data = {
            "text": [
                "I love this idea",
                "This is amazing work",
                "This is terrible and disgusting",
                "You are stupid and useless",
                "Government supports education",
                "Women are not good leaders",
                "All men are dangerous",
                "This group is evil and bad",
                "Education improves society",
                "Violence and hate are increasing",
                "People are kind and helpful",
                "This policy is unfair"
            ],
            "toxicity": [0,0,1,1,0,1,1,1,0,1,0,1],
            "bias": [0,0,1,1,0,1,1,1,0,1,0,1]
        }

        return pd.DataFrame(data)

    # -------------------------
    # TRAIN MODEL
    # -------------------------
    def train(self):

        df = self.load_data()

        X = self.vectorizer.fit_transform(df["text"])

        self.models["toxicity"] = LogisticRegression(max_iter=300)
        self.models["bias"] = LogisticRegression(max_iter=300)

        self.models["toxicity"].fit(X, df["toxicity"])
        self.models["bias"].fit(X, df["bias"])

        print("✔ Safe model trained")

    # -------------------------
    # SENTIMENT
    # -------------------------
    def sentiment(self, text):
        return abs(self.sia.polarity_scores(text)["compound"])

    # -------------------------
    # FRAMING BIAS DETECTOR
    # -------------------------
    def framing(self, text):

        patterns = [
            "are bad", "are evil", "are dangerous",
            "all", "never", "always"
        ]

        t = text.lower()

        return min(sum(0.2 for p in patterns if p in t), 1.0)

    # -------------------------
    # RISK ENGINE
    # -------------------------
    def risk(self, text):

        lexicon = {
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

        return min(sum(v for k,v in lexicon.items() if k in t), 1.0)

    # -------------------------
    # NORMALIZE
    # -------------------------
    def norm(self, x):
        return float(np.clip(x, 0.05, 0.95))

    # -------------------------
    # PREDICT
    # -------------------------
    def predict(self, text):

        X = self.vectorizer.transform([text])

        ml_toxic = self.models["toxicity"].predict_proba(X)[0][1]
        ml_bias = self.models["bias"].predict_proba(X)[0][1]

        risk = self.risk(text)
        sentiment = self.sentiment(text)
        framing = self.framing(text)

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
