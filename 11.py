import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# ========= Linear Regression from scratch (multivariate) =========
def fit_LR_multi(X, y):
    """Normal equation: beta = (XᵀX)⁺ Xᵀy"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    Xb = np.c_[np.ones((len(X), 1)), X]  # add intercept column
    beta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
    return beta.ravel() #ravel 2d to 1d

def predict_LR_multi(X, beta):
    X = np.asarray(X, dtype=float)
    Xb = np.c_[np.ones((len(X), 1)), X]
    return (Xb @ beta).ravel()

# ========= Load & prepare data =========
df = pd.read_csv("Synthetic_House_Price_Dataset.csv").dropna()
df = df[["Area", "Bedrooms", "Location", "Price"]]

# One-hot encode categorical variable
X = pd.get_dummies(df[["Area", "Bedrooms", "Location"]], columns=["Location"], drop_first=True).values
y= df["Price"].values

# ========= K-Fold Cross-Validation =========
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_list, r2_list = [], []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    beta = fit_LR_multi(X_train, y_train)
    y_pred = predict_LR_multi(X_test, beta)
    
    mse_list.append(mean_squared_error(y_test, y_pred))
    r2_list.append(r2_score(y_test, y_pred))

print("=== 5-Fold Cross-Validation ===")
print(f"Average MSE: {np.mean(mse_list):.2f}")
print(f"Average R²: {np.mean(r2_list):.4f}")

# ========= Final model on full dataset =========
beta = fit_LR_multi(X, y)
y_pred_full = predict_LR_multi(X, beta)

print("\n=== Final Model Performance (Full Data) ===")
print(f"MSE: {mean_squared_error(y, y_pred_full):.2f}")
print(f"R²: {r2_score(y, y_pred_full):.4f}")

# ========= Visualization: Actual vs Predicted =========
plt.figure(figsize=(7,6))
plt.scatter(y, y_pred_full, alpha=0.6, label="Predicted vs Actual")
lim = [np.min([y, y_pred_full]), np.max([y, y_pred_full])]

plt.plot(lim, lim, linewidth=2,color='red', label="Ideal Fit")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression (Area + Bedrooms + Location)")
plt.legend()
plt.tight_layout()
plt.show()
