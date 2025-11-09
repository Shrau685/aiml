# Naïve Bayes Spam Detection (train/test + confusion matrix + metrics)
# -------------------------------------------------------------------
# Requires: pandas, scikit-learn, matplotlib
# If needed: pip install pandas scikit-learn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF usually works better than plain counts
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# -----------------------------
# 1) Load & prepare data
# ---------------- -------------
 
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the two useful columns: v1 (label) and v2 (message)
df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"}).dropna()
# Map labels to numbers: ham=0, spam=1
df["y"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)

# -----------------------------
# 2) Train / test split
# -----------------------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"].astype(str), df["y"].values, test_size=0.2, random_state=42, stratify=df["y"].values
)

# -----------------------------
# 3) Vectorize text (TF-IDF)
# -----------------------------
# You can tweak ngram_range/min_df to trade speed vs performance
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X_train = vectorizer.fit_transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)

# -----------------------------
# 4) Train Naïve Bayes
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# 5) Predict & evaluate
# -----------------------------
y_pred = model.predict(X_test)

# acc = accuracy_score(y_test, y_pred)
# prec = precision_score(y_test, y_pred, zero_division=0)
# rec = recall_score(y_test, y_pred, zero_division=0)
# f1 = f1_score(y_test, y_pred, zero_division=0)

y_true = y_test.astype(int)
y_pred = y_pred.astype(int)

TP = int(((y_true == 1) & (y_pred == 1)).sum())
TN = int(((y_true == 0) & (y_pred == 0)).sum())
FP = int(((y_true == 0) & (y_pred == 1)).sum())
FN = int(((y_true == 1) & (y_pred == 0)).sum())

acc  = ((TP + TN)/ (TP + TN + FP + FN))
prec = (TP/ (TP + FP))     # Positive Predictive Value
rec  = (TP/ (TP + FN))     # Sensitivity / TPR
f1   = (2 * prec * rec/prec + rec)

print("=== Naïve Bayes Spam Detection ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# -----------------------------
# 6) Confusion matrix plot
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# Show numbers inside the squares
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.show()


# -----------------------------
# 7) Try one prediction
# -----------------------------
samples = [
    "Congratulations! You have won a free ticket. Call now to claim.",
    "Hey, are we still meeting for lunch at 1?"
]
samples_vec = vectorizer.transform(samples)
preds = model.predict(samples_vec)
for s, p in zip(samples, preds):
    print(f"\nText: {s}\nPredicted: {'spam' if p==1 else 'ham'}")
