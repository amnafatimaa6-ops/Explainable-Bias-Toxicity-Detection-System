import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class BiasModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
        self.models = {}

    def load_data(self):

        data = {
            "text": [
                "I love this idea",
                "This is terrible and disgusting",
                "Government supports education policy",
                "This group is dangerous and bad",
                "Healthcare improves lives",
                "People are angry about decision"
            ],
            "toxicity": [0, 1, 0, 1, 0, 1],
            "identity_attack": [0, 0, 0, 1, 0, 0],
            "insult": [0, 1, 0, 1, 0, 0]
        }

        df = pd.DataFrame(data)

        df["toxicity_label"] = df["toxicity"]
        df["bias_label"] = (
            (df["identity_attack"] == 1) | (df["insult"] == 1)
        ).astype(int)

        return df

    def train(self):

        df = self.load_data()

        X = self.vectorizer.fit_transform(df["text"])

        for target in ["toxicity_label", "bias_label"]:
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)

            self.models[target] = model

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

        return sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:10]
