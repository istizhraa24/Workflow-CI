import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# LOAD DATA

df = pd.read_csv("namadataset_preprocessing/data_clean.csv")

X = df["clean_message"]
y = df["label_encoded"]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer()
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# MLFLOW AUTLOG

mlflow.autolog()

with mlflow.start_run():

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tf, y_train)

    y_pred = model.predict(X_test_tf)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
