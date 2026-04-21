import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score
import os



cols = ["engine_id", "cycle"] + \
       [f"setting_{i}" for i in range(1, 4)] + \
       [f"sensor_{i}" for i in range(1, 22)]

df = pd.read_csv("data/train.csv", sep=" ", header=None)
df = df.dropna(axis=1)
df.columns = cols

print("Data loaded:", df.shape)



max_cycle = df.groupby("engine_id")["cycle"].transform("max")
df["RUL"] = max_cycle - df["cycle"]

# Binary classification: failure within 30 cycles
df["target"] = (df["RUL"] <= 30).astype(int)



# Lag features
df["sensor_1_lag1"] = df.groupby("engine_id")["sensor_1"].shift(1)

# Rolling mean
df["sensor_1_roll_mean"] = (
    df.groupby("engine_id")["sensor_1"]
    .rolling(5)
    .mean()
    .reset_index(0, drop=True)
)

# Drop NaN from lag/rolling
df = df.dropna()

print("After feature engineering:", df.shape)



gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df["engine_id"]))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

X_train = train_df.drop(["target", "RUL"], axis=1)
y_train = train_df["target"]

X_test = test_df.drop(["target", "RUL"], axis=1)
y_test = test_df["target"]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)



model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)



y_prob = model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, y_prob)

print("PR-AUC Score:", pr_auc)



os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.pkl")

print("Model saved at artifacts/model.pkl")
