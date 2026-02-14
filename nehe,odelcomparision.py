import matplotlib.pyplot as plt
import numpy as np

# Models / Papers
models = [
    'Alarfaj et al. (2022)',
    'Hashemi et al. (2022)',
    'Nehe & Devale (2025)',
    'Hybrid Model'
]

# Example literature values (Replace with exact paper values if needed)
precision = [96.0, 79.0, 75.0, 97.71]   # Precision (%)
recall = [91.0, 80.0, 75.0, 86.36]      # Recall (%)

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(x - width/2, precision, width, label='Precision')
plt.bar(x + width/2, recall, width, label='Recall')

plt.xticks(x, models, rotation=10)
plt.ylabel('Percentage (%)')
plt.title('Precision and Recall Comparison Across Fraud Detection Models')
plt.legend()

plt.tight_layout()
plt.savefig('hybrid_vs_literature.png', dpi=300)
plt.show()
