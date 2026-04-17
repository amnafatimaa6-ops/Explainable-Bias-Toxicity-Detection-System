from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class BiasModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.models = {}

    def train(self):

        dataset = load_dataset("civil_comments")
        df = pd.DataFrame(dataset["train"])

        df = df[
            ["text", "toxicity", "identity_attack", "insult"]
        ].dropna()

        df["toxicity_label"] = (df["toxicity"] > 0.5).astype(int)
        df["bias_label"] = (
            (df["identity_attack"] > 0.5) |
            (df["insult"] > 0.7)
        ).astype(int)

        X = self.vectorizer.fit_transform(df["text"])

        for target in ["toxicity_label", "bias_label"]:
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)

            self.models[target] = model

        print("Model trained ✔")

    def predict(self, text):
        vec = self.vectorizer.transform([text])

        return {
            "toxicity": int(self.models["toxicity_label"].predict(vec)[0]),
            "bias_signal": int(self.models["bias_label"].predict(vec)[0])
        }

    def explain(self, text, model_name="toxicity_label"):
        vec = self.vectorizer.transform([text])

        feature_names = self.vectorizer.get_feature_names_out()
        indices = vec.nonzero()[1]
        values = vec.data

        model = self.models[model_name]
        coefs = model.coef_[0]

        scores = []

        for idx, tfidf_val in zip(indices, values):
            word = feature_names[idx]
            score = tfidf_val * coefs[idx]
            scores.append((word, score))

        scores.sort(key=lambda x: abs(x[1]), reverse=True)

        return scores[:10]
