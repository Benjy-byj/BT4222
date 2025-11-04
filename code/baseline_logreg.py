# baseline_logreg.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt
import os

# ---------- Config ----------
EXCEL_PATH = "./../data/Main_Data_Bank.xlsx"           # 改成你的实际路径
SHEET_NAME = "Tilskudd 2025"
TARGET_COL = "Har lån i bank (alle banker)"
NUMERIC_COLS = ["Tilskudd i USD", "Kompleks produksjon (1/0)"]
CATEG_COLS = ["NY tilskuddskategori", "Kommune", "Fylke"]
TEXT_COL = "PRODUKSJON"                      # 半结构化文本（多标签字符串）
OUTPUT_DIR = "baseline_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Step 1. Load ----------
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# ---------- Step 2. Target ----------
# 标准化为小写并转 1/0
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower()
y = (df[TARGET_COL] == "ja").astype(int)

# ---------- Step 3. Features ----------
# 仅使用“公开可得”的特征做baseline
numeric_features = [c for c in NUMERIC_COLS if c in df.columns]
categorical_features = [c for c in CATEG_COLS if c in df.columns]
text_feature = TEXT_COL if TEXT_COL in df.columns else None
if text_feature is None:
    text_feature = "__EMPTY_TEXT__"
    df[text_feature] = ""

# 强制数值列为数值，出错转NaN
for c in numeric_features:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 缺失填充
X = df[numeric_features + categorical_features + [text_feature]].copy()
for c in numeric_features:
    X[c] = X[c].fillna(X[c].median())
for c in categorical_features:
    X[c] = X[c].astype(str).fillna("")
X[text_feature] = X[text_feature].astype(str).fillna("")

# ---------- Step 4. Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Step 5. Preprocess ----------
transformers = []
if numeric_features:
    transformers.append(("num", StandardScaler(), numeric_features))
if categorical_features:
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))
# 文本用TF-IDF（给一个较小上限以保持Baseline简洁）
text_transformer = ("txt", TfidfVectorizer(max_features=100), text_feature)

preprocess = ColumnTransformer(
    transformers=transformers + [text_transformer],
    remainder="drop",
    verbose_feature_names_out=True,
)

# ---------- Step 6. Model ----------
# 如果类不平衡明显，可以把 class_weight="balanced" 打开做对比
clf = LogisticRegression(max_iter=1000)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", clf),
])

pipe.fit(X_train, y_train)

# ---------- Step 7. Evaluate ----------
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
metrics_path = os.path.join(OUTPUT_DIR, "baseline_logreg_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print("== Baseline Metrics ==")
print(metrics_df)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ROC曲线（可视化）
fig_roc, ax_roc = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc)
ax_roc.set_title("ROC Curve (Baseline Logistic Regression)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)

# ---------- Step 8. Coefficients (TopK) ----------
feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
coefs = pipe.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs,
    "abs_coef": np.abs(coefs),
}).sort_values("abs_coef", ascending=False)

top_k = 25
top_coef_df = coef_df.head(top_k).drop(columns=["abs_coef"]).round(4)
coef_path = os.path.join(OUTPUT_DIR, "baseline_logreg_top25_coeffs.csv")
top_coef_df.to_csv(coef_path, index=False)

print(f"\nSaved metrics to: {metrics_path}")
print(f"Saved top coefficients to: {coef_path}")
print(f"Saved ROC curve to: {os.path.join(OUTPUT_DIR, 'roc_curve.png')}")
