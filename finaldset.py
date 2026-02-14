import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# --- Load dataset ---
df = pd.read_csv("creditcard.csv")

print("🔹 Original dataset shape:", df.shape)
print("🔹 Original class distribution:")
print(df["Class"].value_counts())

# --- Step 1: Downsample majority class ---
fraud_df = df[df["Class"] == 1]
nonfraud_df = df[df["Class"] == 0]

nonfraud_down = nonfraud_df.sample(frac=0.4, random_state=42)

df_balanced = pd.concat([fraud_df, nonfraud_down]) \
                .sample(frac=1, random_state=42) \
                .reset_index(drop=True)

print("\n✅ After downsampling:")
print(df_balanced["Class"].value_counts())

# --- Step 2: Separate features & target ---
X = df_balanced.drop(columns=["Class"])
y = df_balanced["Class"]

print("\n📊 Class distribution BEFORE SMOTE:")
print(y.value_counts())

# Save Time column
time_col = X["Time"].values
X_features = X.drop(columns=["Time"])

# --- Step 3: Compute desired fraud count for 70:30 ratio ---
nonfraud_count = y.value_counts()[0]

desired_fraud_count = int((30 / 70) * nonfraud_count)

print("\n🎯 Target fraud samples for 70:30 ratio:", desired_fraud_count)

# --- Step 4: Apply SMOTE ---
sm = SMOTE(
    sampling_strategy={1: desired_fraud_count},
    random_state=42
)

X_res, y_res = sm.fit_resample(X_features, y)

print("\n📊 Class distribution AFTER SMOTE:")
print(pd.Series(y_res).value_counts())

# --- Step 5: Rebuild DataFrame ---
df_aug = pd.DataFrame(X_res, columns=X_features.columns)
df_aug["Class"] = y_res

# --- Step 6: Recreate Time column ---
n_new = len(df_aug) - len(df_balanced)

time_min = df_balanced["Time"].min()
time_max = df_balanced["Time"].max()

time_aug = np.concatenate([
    time_col,
    np.random.randint(time_min, time_max + 1, n_new)
])

df_aug["Time"] = time_aug

# Reorder columns
cols = ["Time"] + [c for c in df_aug.columns if c != "Time"]
df_aug = df_aug[cols]

print("\n✅ Final dataset:")
print(df_aug["Class"].value_counts())

# --- Step 7: Save dataset ---
df_aug.to_csv("finaldataset.csv", index=False)

print("\n📂 Saved as finaldataset.csv")
