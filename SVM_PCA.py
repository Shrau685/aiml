import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Load dataset
data = pd.read_csv("/mnt/data/emails_16_17_18_19.csv")

# Optional: reduce data for faster processing
data = data.sample(n=2000, random_state=42)

# Prepare features and labels
X = data.drop(columns=["Email No.", "Prediction"])
y = data["Prediction"].astype(int)

# Balance data using undersampling
data_balanced = pd.concat([X, y], axis=1)
majority = data_balanced[data_balanced["Prediction"] == 0]
minority = data_balanced[data_balanced["Prediction"] == 1]
majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
balanced_data = pd.concat([majority_downsampled, minority])

X_bal = balanced_data.drop(columns=["Prediction"])
y_bal = balanced_data["Prediction"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)

# -----------------------
# Evaluation Metrics
# -----------------------
TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

print("\n Model Performance:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# -----------------------
# PCA for Visualization
# -----------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_scaled)

# Plot scatter plot with hyperplane
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y_pred == 0, 0], X_pca[y_pred == 0, 1], color='purple', label='Not Spam', alpha=0.6)
plt.scatter(X_pca[y_pred == 1, 0], X_pca[y_pred == 1, 1], color='orange', label='Spam', alpha=0.6)

# Compute SVM decision boundary (in PCA space)
w = svm_model.coef_[0]
b = svm_model.intercept_[0]
# Approximate hyperplane in PCA 2D projection
x_plot = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
y_plot = -(w[0]/w[1])*x_plot - b/w[1]
plt.plot(x_plot, y_plot, 'k--', label='Hyperplane')

plt.title("SVM Spam Detection: Predicted Classes & Hyperplane")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# -----------------------
# User Input Prediction
# -----------------------
print("\n User Input Section:")
try:
    # Create a simple example: user provides some word counts
    example = {
        "free": int(input("Enter count of word 'free': ")),
        "money": int(input("Enter count of word 'money': ")),
        "click": int(input("Enter count of word 'click': ")),
        "offer": int(input("Enter count of word 'offer': ")),
        "you": int(input("Enter count of word 'you': ")),
    }

    # Match input to dataset columns (fill others with 0)
    user_df = pd.DataFrame([np.zeros(X.shape[1])], columns=X.columns)
    for word, val in example.items():
        if word in user_df.columns:
            user_df[word] = val

    user_scaled = scaler.transform(user_df)
    pred = svm_model.predict(user_scaled)[0]

    if pred == 1:
        print(" The email is predicted as: SPAM")
    else:
        print("The email is predicted as: NOT SPAM")

except Exception as e:
    print(" Input error:", e)
