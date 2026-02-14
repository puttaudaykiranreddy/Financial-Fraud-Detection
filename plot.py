import matplotlib.pyplot as plt
import numpy as np

# Metrics dictionary
metrics = {
    "Random Forest": {'Accuracy': 0.99, 'Precision': 0.9418604651162791, 'Recall': 0.826530612244898,
                      'F1': 0.8804347826086957, 'ROC_AUC': 0.9528106983508091},
    "DFNN": {'Accuracy': 0.99, 'Precision': 0.8295454545454546, 'Recall': 0.7448979591836735,
             'F1': 0.7849462365591398, 'ROC_AUC': 0.966258376592055},
    "Combined": {'Accuracy': 0.99, 'Precision': 0.96, 'Recall': 0.7346938775510204,
                 'F1': 0.8323699421965318, 'ROC_AUC': 0.9692246376603539}
}

# Prepare data
models = list(metrics.keys())
metric_names = list(next(iter(metrics.values())).keys())
values = [[metrics[m][met] for m in models] for met in metric_names]

x = np.arange(len(models))
width = 0.15

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for i, (met, vals) in enumerate(zip(metric_names, values)):
    rects = ax.bar(x + i*width - (width*len(metric_names)/2), vals, width, label=met)
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Labels & formatting
ax.set_ylabel('Score')
ax.set_title('Model Performance Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()

# Save as vector graphic
plt.savefig("model_performance_metrics.svg", format="svg")
plt.savefig("model_performance_metrics.pdf", format="pdf")

plt.show()
