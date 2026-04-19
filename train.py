import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score

# Load dataset
df = pd.read_csv("data/train.csv")

# Create RUL
max_cycle = df.groupby("engine_id")["cycle"].transform("max")
df["RUL"] = max_cycle - df["cycle"]

# Create target (failure within 30 cycles)
df["target"] = (df["RUL"] <= 30).astype(int)

# Feature engineering
df["sensor_1_lag1"] = df.groupby("engine_id")["sensor_1"].shift(1)
df["sensor_1_roll_mean"] = df.groupby("engine_id")["sensor_1"].rolling(5).mean().reset_index(0, drop=True)

df = df.dropna()

# Split (IMPORTANT: group by engine_id)
gss = GroupShuffleSplit(test_size=0.2, n_splits=1)
train_idx, test_idx = next(gss.split(df, groups=df["engine_id"]))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

X_train = train_df.drop(["target", "RUL"], axis=1)
y_train = train_df["target"]

X_test = test_df.drop(["target", "RUL"], axis=1)
y_test = test_df["target"]

# Model
model = LGBMClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Evaluation
y_prob = model.predict_proba(X_test)[:, 1]
score = average_precision_score(y_test, y_prob)
print("PR-AUC:", score)

# Save model
joblib.dump(model, "artifacts/model.pkl")
