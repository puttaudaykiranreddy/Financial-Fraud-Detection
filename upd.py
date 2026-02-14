import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Load Dataset ---
df = pd.read_csv("creditcard_augmented.csv")

# Features and target
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- Load or Train Models ---
if os.path.exists("rf_model.pkl") and os.path.exists("dfnn_model.h5"):
    print("✅ Loading saved models...")
    rf = joblib.load("rf_model.pkl")
    dfnn = load_model("dfnn_model.h5")
else:
    print("⏳ Training new models...")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "rf_model.pkl")
    print("✅ RF model saved as rf_model.pkl")

    # DFNN
    dfnn = models.Sequential([
        layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    dfnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    dfnn.fit(X_train, y_train, epochs=5, batch_size=2048, verbose=1)
    dfnn.save("dfnn_model.h5")
    print("✅ DFNN model saved as dfnn_model.h5")

# --- Threshold Settings ---
threshold_rf = 0.45
threshold_dfnn = 0.45
threshold_hybrid = 0.25

# --- Predictions ---
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = (y_prob_rf > threshold_rf).astype(int)

y_prob_dfnn = dfnn.predict(X_test).ravel()
y_pred_dfnn = (y_prob_dfnn > threshold_dfnn).astype("int32")

y_prob_combined = (y_prob_rf + y_prob_dfnn) / 2
y_pred_combined = (y_prob_combined > threshold_hybrid).astype(int)
print("✅ Thresholds applied internally: RF=0.4, DFNN=0.4, Hybrid=0.4")

# --- Metrics Function ---
def get_metrics(y_true, y_pred, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc
    }

metrics_rf = get_metrics(y_test, y_pred_rf, y_prob_rf)
metrics_dfnn = get_metrics(y_test, y_pred_dfnn, y_prob_dfnn)
metrics_combined = get_metrics(y_test, y_pred_combined, y_prob_combined)

# --- Print Metrics ---
print("\n📊 Performance Metrics:")
print("Random Forest:", metrics_rf)
print("DFNN:", metrics_dfnn)
print("Hybrid:", metrics_combined)

# --- Plot Metrics Bar Chart  ---
labels = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
rf_scores = [metrics_rf[label] for label in labels]
dfnn_scores = [metrics_dfnn[label] for label in labels]
combined_scores = [metrics_combined[label] for label in labels]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(10,6))
bars1 = plt.bar(x - width, rf_scores, width, label="RF")
bars2 = plt.bar(x, dfnn_scores, width, label="DFNN")
bars3 = plt.bar(x + width, combined_scores, width, label="Hybrid")

for i, bar in enumerate(bars1):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{rf_scores[i]:.2f}", ha='center', va='bottom', fontweight='bold')
for i, bar in enumerate(bars2):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{dfnn_scores[i]:.2f}", ha='center', va='bottom', fontweight='bold')
for i, bar in enumerate(bars3):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{combined_scores[i]:.2f}", ha='center', va='bottom', fontweight='bold')

plt.ylabel("Score", fontweight="bold")
plt.title("Performance Metrics Comparison", fontweight="bold")
plt.xticks(x, labels, fontweight="bold")
plt.ylim(0,1.15)
plt.legend()
plt.tight_layout()
plt.savefig("performance_metrics.pdf", bbox_inches="tight")
plt.savefig("performance_metrics.svg", bbox_inches="tight")
plt.close()
print("📂 Metrics bar chart saved (PDF & SVG)")

# --- ROC Curves  ---
def plot_roc(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1], [0,1], color='red', linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title(f"ROC Curve - {model_name}", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"roc_curve_{model_name.lower().replace(' ','_')}.pdf", bbox_inches="tight")
    plt.close()

plot_roc(y_test, y_prob_rf, "Random Forest")
plot_roc(y_test, y_prob_dfnn, "DFNN")
plot_roc(y_test, y_prob_combined, "Hybrid")
print("📂 ROC curves saved as PDFs")

# --- Save Fraud and Non-Fraud Accounts ---
fraud_cases = df[df["Class"] == 1]
nonfraud_cases = df[df["Class"] == 0]

fraud_cases.to_csv("fraud_accounts.csv", index=False)
nonfraud_cases.to_csv("nonfraud_accounts.csv", index=False)
print("📂 Fraud and Non-fraud CSVs saved")
