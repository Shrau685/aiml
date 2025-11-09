# ------------------ Import Libraries ------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

# ------------------ Load Dataset ------------------
df = pd.read_csv("disease_diagnosis_16_17.csv")

# ------------------ Preprocessing ------------------
# Drop columns not required for classification
df_clean = df.drop(columns=['Patient_ID', 'Treatment_Plan', 'Severity'])

# Encode categorical columns
le = LabelEncoder()
for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col] = le.fit_transform(df_clean[col])

# Split into features and labels
X = df_clean.drop(columns=['Diagnosis'])
y = df_clean['Diagnosis']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Naïve Bayes (From Scratch) ------------------
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-6  # add small value to avoid division by zero
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            conditional = np.sum(np.log(self._pdf(c, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict(x) for x in X.to_numpy()])

# ------------------ Train Model ------------------
nb = NaiveBayes()
nb.fit(X_train, y_train)

# ------------------ Predict ------------------
y_pred = nb.predict(X_test)

# ------------------ Evaluation ------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ------------------ Confusion Matrix ------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Naïve Bayes (From Scratch)")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()

# ------------------ Sample Predictions ------------------
results = pd.DataFrame({
    'Actual': y_test[:10].values,
    'Predicted': y_pred[:10]
})
print("\nSample Predictions:")
print(results)