import os
import re
import numpy as np
import pandas as pd
import pm4py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
import xgboost as xgb
import shap

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Paths / config
# ---------------------------
xes_path = "data/BPI_Challenge_2019-3-w-after.xes"
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

SHAP_MAX_SAMPLES = 5000  # limit rows for plotting speed

# ---------------------------
# Helpers: sanitize and normalize names
# ---------------------------
_invalid_re = re.compile(r"[\[\]<>]")
_space_re = re.compile(r"\s+")

def sanitize_feature_names(cols):
    out = []
    seen = {}
    for c in map(str, cols):
        name = _invalid_re.sub("_", c)
        name = _space_re.sub("_", name)
        base = name
        k = seen.get(base, 0)
        if k > 0:
            name = f"{base}__dup{k}"
        seen[base] = k + 1
        out.append(name)
    return out

def sanitize_columns(df):
    df = df.copy()
    df.columns = sanitize_feature_names(df.columns)
    return df

def canonicalize(name: str) -> str:
    s = name.lower()
    s = s.replace("case:", "")
    s = s.replace("(case)", "")
    s = s.strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def resolve_case_columns(df, requested_labels):
    canon_to_display = {canon: display for display, canon in requested_labels.items()}
    found = {}
    for col in df.columns:
        c = canonicalize(col)
        if c in canon_to_display and c not in found:
            found[c] = col
    resolved = {}
    for canon, col in found.items():
        display = canon_to_display[canon]
        resolved[display] = col
    return resolved

# ---------------------------
# Load XES -> DataFrame
# ---------------------------
obj = pm4py.read_xes(xes_path)
df = obj if isinstance(obj, pd.DataFrame) else pm4py.convert_to_dataframe(obj)

CASE_ID = "case:concept:name"
TS = "time:timestamp"

# Timestamps
df[TS] = pd.to_datetime(df[TS], errors="coerce", utc=True)

# ---------------------------
# Target: case duration (days) with hard cutoff at 400
# ---------------------------
case_bounds = df.groupby(CASE_ID)[TS].agg(case_start="min", case_end="max")
case_bounds["case_duration"] = case_bounds["case_end"] - case_bounds["case_start"]
case_bounds["duration_days"] = case_bounds["case_duration"].dt.total_seconds() / (3600 * 24)
case_bounds["duration_days"] = case_bounds["duration_days"].clip(upper=400)  # Changed to 400

plt.figure(figsize=(10, 6))
sns.histplot(case_bounds["duration_days"], bins=100, kde=True)
plt.title("Distribution of Case Duration (in days)")
plt.xlabel("Duration (days)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "duration_distribution.png"))
plt.close()

# ---------------------------
# Case-only features: EXACTLY these five attributes
# ---------------------------
requested = {
    "Item": "item",
    "Item Type": "itemtype",
    "Vendor": "vendor",
    "Spend area text": "spendareatext",
    "Sub spend area text": "subspendareatext",
}

resolved = resolve_case_columns(df, requested)
missing = [disp for disp in requested if disp not in resolved]
if missing:
    sample = [c for c in df.columns if ("case" in c.lower()) or ("(case" in c.lower())]
    raise ValueError(
        f"Requested attributes not found: {missing}. "
        f"Found candidate case columns (sample): {sample[:20]}"
    )

selected_cols = [resolved[k] for k in requested.keys()]

# Aggregate case-level values
case_static = (
    df.sort_values(TS)
      .groupby(CASE_ID)[selected_cols]
      .first()
)

# Add temporal features from case start timestamp
case_static = case_static.join(case_bounds["case_start"])
case_static["start_year"] = case_static["case_start"].dt.year
case_static["start_month"] = case_static["case_start"].dt.month
case_static["start_day_of_week"] = case_static["case_start"].dt.dayofweek
case_static["start_hour"] = case_static["case_start"].dt.hour
case_static = case_static.drop(columns=["case_start"])

# Join target
data = case_static.join(case_bounds[["duration_days"]], how="inner")
data.index = data.index.rename("case_id")
data = data.reset_index()

# ---------------------------
# Split + preprocessing
# ---------------------------
feature_cols = [c for c in data.columns if c not in ["case_id", "duration_days"]]

def is_bool_col(s):
    return pd.api.types.is_bool_dtype(s) or str(s.dtype) in ("bool", "boolean")

cat_cols = [c for c in feature_cols if (data[c].dtype == "object") or is_bool_col(data[c])]
num_cols = [c for c in feature_cols if c not in cat_cols]

pre = ColumnTransformer(
    transformers=[
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
        ("num", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median"))
        ]), num_cols),
    ],
    remainder="drop",
)

X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
y_train = X_train["duration_days"].astype(float)
y_test = X_test["duration_days"].astype(float)

X_tr, X_val, y_tr, y_val = train_test_split(X_train[feature_cols], y_train, test_size=0.2, random_state=42)

pre = pre.set_output(transform="pandas")
pre.fit(X_tr)

X_tr_enc = sanitize_columns(pre.transform(X_tr))
X_val_enc = sanitize_columns(pre.transform(X_val))
X_test_enc = sanitize_columns(pre.transform(X_test[feature_cols]))

if X_tr_enc.shape[1] == 0:
    raise ValueError("Preprocessing produced 0 feature columns; check attribute selection and imputers.")

def sanitized_base(colname):
    return sanitize_feature_names([colname])[0]

base_map = {}
for display, orig in resolved.items():
    base_map[sanitized_base(orig)] = display

# ---------------------------
# XGBoost with tuned hyperparameters
# ---------------------------
reg = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=3000,           # Increased
    learning_rate=0.05,          # Slightly higher for faster convergence
    max_depth=6,                 # Reduced to prevent overfitting
    min_child_weight=3,          # Added regularization
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,                   # Added regularization
    reg_alpha=0.1,               # L1 regularization
    reg_lambda=1.0,              # L2 regularization
    random_state=42,
    n_jobs=8,
    early_stopping_rounds=50,
)

reg.fit(
    X_tr_enc, y_tr,
    eval_set=[(X_val_enc, y_val)],
    verbose=False
)

preds = reg.predict(X_test_enc)
mae = mean_absolute_error(y_test, preds)
print(f"MAE (days): {mae:.4f}")

# ---------------------------
# SHAP aggregation
# ---------------------------
if len(X_test_enc) > SHAP_MAX_SAMPLES:
    X_shap = X_test_enc.sample(SHAP_MAX_SAMPLES, random_state=42)
else:
    X_shap = X_test_enc

print(f"Computing SHAP on {len(X_shap)} rows using XGBoost pred_contribs ...")

dtest = xgb.DMatrix(X_shap, feature_names=X_shap.columns.astype(str).tolist())
contribs = reg.get_booster().predict(dtest, pred_contribs=True)
shap_values = contribs[:, :-1]
expected_value = contribs[:, -1].mean()

ex = shap.Explanation(
    values=shap_values,
    base_values=np.full(X_shap.shape[0], expected_value),
    data=X_shap.values,
    feature_names=X_shap.columns.astype(str).tolist(),
)

shap.plots.violin(ex, show=False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "shap_summary_violin.png"), dpi=150)
plt.close()

shap.plots.bar(ex, show=False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "shap_summary_bar.png"), dpi=150)
plt.close()

col_names_str = X_shap.columns.astype(str).tolist()
mean_abs_by_col = np.abs(shap_values).mean(axis=0)

group_scores = {}
for base_sanitized, display in base_map.items():
    cat_prefix = f"cat__{base_sanitized}_"
    num_name = f"num__{base_sanitized}"
    idxs = [i for i, name in enumerate(col_names_str) if name.startswith(cat_prefix) or name == num_name]
    group_scores[display] = float(mean_abs_by_col[idxs].sum()) if len(idxs) > 0 else 0.0

group_df = pd.DataFrame({
    "feature": list(group_scores.keys()),
    "mean_abs_shap_group": list(group_scores.values())
}).sort_values("mean_abs_shap_group", ascending=False)

group_df.to_csv(os.path.join(out_dir, "shap_feature_groups.csv"), index=False)

plt.figure(figsize=(8, 4))
plt.barh(group_df["feature"], group_df["mean_abs_shap_group"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP| (aggregated)")
plt.title("Global impact by original attribute")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "shap_feature_groups_bar.png"), dpi=150)
plt.close()

print("Saved:")
print(f"- {os.path.join(out_dir, 'duration_distribution.png')}")
print(f"- {os.path.join(out_dir, 'shap_summary_violin.png')}")
print(f"- {os.path.join(out_dir, 'shap_summary_bar.png')}")
print(f"- {os.path.join(out_dir, 'shap_feature_groups.csv')}")
print(f"- {os.path.join(out_dir, 'shap_feature_groups_bar.png')}")
