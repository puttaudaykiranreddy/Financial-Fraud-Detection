import matplotlib.pyplot as plt
import numpy as np

# --- Exact Metrics Provided ---
metrics_rf = {
    'Accuracy': 0.99,
    'Precision': 0.9823529411764705,
    'Recall': 0.8434343434343434,
    'F1': 0.907608695652174,
    'ROC_AUC': 0.9841523917625211
}

metrics_dfnn = {
    'Accuracy': 0.996,
    'Precision': 0.975609756097561,
    'Recall': 0.8080808080808081,
    'F1': 0.8839779005524862,
    'ROC_AUC': 0.9757208948715148
}

metrics_hybrid = {
    'Accuracy': 0.99637378,
    'Precision': 0.9763313609467456,
    'Recall': 0.8333333333333334,
    'F1': 0.8991825613079019,
    'ROC_AUC': 0.99
}

labels = list(metrics_rf.keys())
rf_scores = [metrics_rf[label] for label in labels]
dfnn_scores = [metrics_dfnn[label] for label in labels]
hybrid_scores = [metrics_hybrid[label] for label in labels]

x = np.arange(len(labels))
width = 0.25

# --- Plot ---
plt.figure(figsize=(10,6))
bars1 = plt.bar(x - width, rf_scores, width, label="Random Forest")
bars2 = plt.bar(x, dfnn_scores, width, label="DFNN")
bars3 = plt.bar(x + width, hybrid_scores, width, label="Hybrid")

# Values on top of bars (2 decimal points)
fontsize_values = 8
for i, bar in enumerate(bars1):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{rf_scores[i]:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=fontsize_values)
for i, bar in enumerate(bars2):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{dfnn_scores[i]:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=fontsize_values)
for i, bar in enumerate(bars3):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{hybrid_scores[i]:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=fontsize_values)

# Side labels fixed at left margin
for i, label in enumerate(labels):
    plt.text(-0.7, x[i], label, ha='right', va='center', fontweight='bold', fontsize=9)

plt.ylabel("Score", fontweight="bold")
plt.title("Performance Metrics Comparison", fontweight="bold")
plt.xticks(x, labels, fontweight="bold")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

# Save as vector files
plt.savefig("metrics_comparison.pdf", bbox_inches="tight")
plt.savefig("metrics_comparison.svg", bbox_inches="tight")
plt.show()


