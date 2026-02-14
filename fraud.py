import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# --- Load dataset (with Time) ---
df = pd.read_csv("creditcard.csv")

print("🔹 Original dataset shape:", df.shape)
print("🔹 Class distribution before:")
print(df["Class"].value_counts())

# --- Step 1: Downsample non-fraud (Class=0) by 60% ---
fraud_df = df[df["Class"] == 1]
nonfraud_df = df[df["Class"] == 0]

nonfraud_down = nonfraud_df.sample(frac=0.4, random_state=42)  # keep 40%

df_balanced = pd.concat([fraud_df, nonfraud_down]).sample(frac=1, random_state=42).reset_index(drop=True)

print("\n✅ After downsampling:")
print("🔹 Shape:", df_balanced.shape)
print("🔹 Class distribution:")
print(df_balanced["Class"].value_counts())

# --- Step 2: Separate features and target ---
X = df_balanced.drop(columns=["Class"])
y = df_balanced["Class"]

# Save Time column separately
time_col = X["Time"].values
X_features = X.drop(columns=["Time"])  # SMOTE only on numeric features

# --- Step 3: Apply SMOTE (add 500 synthetic frauds) ---
sm = SMOTE(sampling_strategy={1: y.value_counts()[1] + 500}, random_state=42)
X_res, y_res = sm.fit_resample(X_features, y)

# --- Step 4: Recreate DataFrame ---
df_aug = pd.DataFrame(X_res, columns=X_features.columns)
df_aug["Class"] = y_res

# --- Step 5: Generate Time column for synthetic rows ---
n_new = len(df_aug) - len(df_balanced)
time_min, time_max = df_balanced["Time"].min(), df_balanced["Time"].max()

time_aug = np.concatenate([
    time_col,  # original times
    np.random.randint(time_min, time_max + 1, n_new)  # random times for synthetic
])

df_aug["Time"] = time_aug

# Reorder columns to match original
cols = ["Time"] + [c for c in df_aug.columns if c not in ["Time"]]
df_aug = df_aug[cols]

print("\n✅ After augmentation (with downsampling + SMOTE):")
print("🔹 Final dataset shape:", df_aug.shape)
print("🔹 Class distribution after:")
print(df_aug["Class"].value_counts())

# --- Step 6: Save new dataset ---
df_aug.to_csv("creditcard_augmented.csv", index=False)
print("\n📂 Saved as creditcard_augmented.csv")
