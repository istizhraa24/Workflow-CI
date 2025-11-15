import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import argparse

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

data_path = args.data_path
print("Dataset path:", data_path)

# LOAD DATA
df = pd.read_csv(data_path)

# FIX: Bersihkan NaN di teks

if "clean_message" not in df.columns:
    raise ValueError("Kolom 'clean_message' tidak ditemukan dalam dataset!")

df = df.dropna(subset=["clean_message"])            # hapus NaN
df["clean_message"] = df["clean_message"].astype(str)  # ubah ke string
df = df[df["clean_message"].str.strip() != ""]      # buang string kosong

# Pastikan label juga valid
df = df.dropna(subset=["label_encoded"])

# FEATURES & LABEL
X = df["clean_message"]
y = df["label_encoded"]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# LOGGING MLFLOW
mlflow.log_metric("accuracy", acc)
