# SVM Spam Detection with Imbalance Handling (oversampling) on spam.csv
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

# 1) Load

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1","v2"]].rename(columns={"v1":"label","v2":"text"}).dropna()
df["y"] = df["label"].map({"ham":0, "spam":1}).astype(int)

# 2) Split (stratified)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"].astype(str), df["y"].values, test_size=0.2, random_state=42, stratify=df["y"].values
)

# 3) TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X_train = vectorizer.fit_transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)

# 4) Random oversampling on TRAIN set only (manual)
# def random_oversample_sparse(X, y, seed=42):
#     rng = np.random.default_rng(seed)
#     y = np.asarray(y)
#     classes, counts = np.unique(y, return_counts=True)
#     max_count = counts.max()
#     idx = np.arange(len(y))
#     new_idx = []
#     for c in classes:
#         c_idx = idx[y == c]
#         add = rng.choice(c_idx, size=max_count - len(c_idx), replace=True) if len(c_idx) < max_count else np.array([], dtype=int)
#         new_idx.append(np.concatenate([c_idx, add]))
#     new_idx = np.concatenate(new_idx)
#     rng.shuffle(new_idx)
#     return X[new_idx], y[new_idx]

# X_train_bal, y_train_bal = random_oversample_sparse(X_train, y_train, seed=42)

# 5) Train SVM
model = LinearSVC(C=1.0, random_state=42)
model.fit(X_train, y_train)

# 6) Evaluate
y_pred = model.predict(X_test)

y_true = y_test.astype(int)
y_pred = y_pred.astype(int)

TP = int(((y_true == 1) & (y_pred == 1)).sum())
TN = int(((y_true == 0) & (y_pred == 0)).sum())
FP = int(((y_true == 0) & (y_pred == 1)).sum())
FN = int(((y_true == 1) & (y_pred == 0)).sum())

acc  = ((TP + TN)/ (TP + TN + FP + FN))
prec = (TP/ (TP + FP))     # Positive Predictive Value
rec  = (TP/(TP + FN))     # Sensitivity / TPR
f1 = f1_score(y_test, y_pred)



print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["ham","spam"]))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix [[TN, FP],[FN, TP]]:\n", cm)



plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# Show numbers inside the squares
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.show()




# # ROC curve
# fpr, tpr, _ = roc_curve(y_test, scores)
# plt.figure(figsize=(5.5,4.5))
# plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
# plt.plot([0,1], [0,1], linestyle="--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve - SVM (Oversampling)")
# plt.legend()
# plt.tight_layout()
# plt.show()
