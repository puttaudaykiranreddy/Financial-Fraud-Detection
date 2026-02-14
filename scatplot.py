import pandas as pd
import matplotlib.pyplot as plt

# --- Load datasets ---
# BEFORE = downsampled dataset (create it again or load if saved)
df_original = pd.read_csv("creditcard.csv")

# recreate downsampled dataset (same logic as before)
fraud_df = df_original[df_original["Class"] == 1]
nonfraud_df = df_original[df_original["Class"] == 0]
nonfraud_down = nonfraud_df.sample(frac=0.4, random_state=42)

df_balanced = pd.concat([fraud_df, nonfraud_down]).sample(
    frac=1, random_state=42
)

# AFTER = augmented dataset
df_aug = pd.read_csv("creditcard_augmented.csv")

# --- Choose features ---
f1 = "V1"
f2 = "V2"

# Split classes
before_nonfraud = df_balanced[df_balanced["Class"] == 0]
before_fraud = df_balanced[df_balanced["Class"] == 1]

after_nonfraud = df_aug[df_aug["Class"] == 0]
after_fraud = df_aug[df_aug["Class"] == 1]

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BEFORE
axes[0].scatter(before_nonfraud[f1], before_nonfraud[f2],
                alpha=0.3, label="Non-Fraud", s=10)

axes[0].scatter(before_fraud[f1], before_fraud[f2],
                alpha=0.9, label="Fraud", marker="x", s=30)

axes[0].set_title("A — Before SMOTE")
axes[0].legend()

# AFTER
axes[1].scatter(after_nonfraud[f1], after_nonfraud[f2],
                alpha=0.3, label="Non-Fraud", s=10)

axes[1].scatter(after_fraud[f1], after_fraud[f2],
                alpha=0.9, label="Fraud", marker="x", s=30)

axes[1].set_title("B — After SMOTE")
axes[1].legend()

plt.tight_layout()
plt.show()
