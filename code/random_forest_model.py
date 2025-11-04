# random_forest_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt
import os

# ---------- Config ----------
EXCEL_PATH = "./../data/Main_Data_Bank.xlsx"   # 修改为你的路径
SHEET_NAME = "Tilskudd 2025"
TARGET_COL = "Har lån i bank (alle banker)"
NUMERIC_COLS = ["Tilskudd i USD", "Kompleks produksjon (1/0)"]
CATEG_COLS = ["NY tilskuddskategori", "Kommune", "Fylke"]
TEXT_COL = "PRODUKSJON"
OUTPUT_DIR = "rf_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Step 1: Load ----------
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# ---------- Step 2: Target ----------
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower()
y = (df[TARGET_COL] == "ja").astype(int)

# ---------- Step 3: Features ----------
numeric_features = [c for c in NUMERIC_COLS if c in df.columns]
categorical_features = [c for c in CATEG_COLS if c in df.columns]
text_feature = TEXT_COL if TEXT_COL in df.columns else None
if text_feature is None:
    text_feature = "__EMPTY_TEXT__"
    df[text_feature] = ""

for c in numeric_features:
    df[c] = pd.to_numeric(df[c], errors="coerce")

X = df[numeric_features + categorical_features + [text_feature]].copy()

for c in numeric_features:
    X[c] = X[c].fillna(X[c].median())
for c in categorical_features:
    X[c] = X[c].astype(str).fillna("")
X[text_feature] = X[text_feature].astype(str).fillna("")

# ---------- Step 4: Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Step 5: Preprocessing ----------
transformers = []
if numeric_features:
    transformers.append(("num", StandardScaler(), numeric_features))
if categorical_features:
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))
text_transformer = ("txt", TfidfVectorizer(max_features=100), text_feature)

preprocess = ColumnTransformer(
    transformers=transformers + [text_transformer],
    remainder="drop",
    verbose_feature_names_out=True,
)

# ---------- Step 6: Model ----------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("rf", rf),
])

# ---------- Step 7: Train ----------
pipe.fit(X_train, y_train)

# ---------- Step 8: Evaluate ----------
y_prob = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1": f1_score(y_test, y_pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, y_prob),
}
metrics_df = pd.DataFrame([metrics]).round(4)
metrics_path = os.path.join(OUTPUT_DIR, "rf_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print("== Random Forest Metrics ==")
print(metrics_df)

# ---------- Step 9: ROC Curve ----------
fig_roc, ax_roc = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc)
ax_roc.set_title("ROC Curve (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_roc_curve.png"), dpi=150)

# ---------- Step 10: Feature Importance ----------
feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
importances = pipe.named_steps["rf"].feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False)

top_k = 25
top_fi_df = fi_df.head(top_k).round(4)
top_path = os.path.join(OUTPUT_DIR, "rf_top25_features.csv")
top_fi_df.to_csv(top_path, index=False)

plt.figure(figsize=(8, 6))
plt.barh(top_fi_df["feature"][::-1], top_fi_df["importance"][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 25 Important Features (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_feature_importance.png"), dpi=150)

print(f"\nSaved metrics to: {metrics_path}")
print(f"Saved top features to: {top_path}")
print(f"Saved ROC curve to: {os.path.join(OUTPUT_DIR, 'rf_roc_curve.png')}")
print(f"Saved importance plot to: {os.path.join(OUTPUT_DIR, 'rf_feature_importance.png')}")
