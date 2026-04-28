# train.py — Customer Churn Prediction Training
# Run: python train.py

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

BANNER = "=" * 55
print(BANNER)
print("   Customer Churn Prediction — Model Training")
print(BANNER)

# ── 1. Load Data ──────────────────────────────────────────
csv_path = "cleaned_telco_churn.csv"
if not os.path.exists(csv_path):
    print(f"[ERROR] '{csv_path}' not found.")
    exit(1)

df = pd.read_csv(csv_path)
print(f"[INFO] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# ── 2. Clean ──────────────────────────────────────────────
df.drop(columns=["customerID"], errors="ignore", inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
print(f"[INFO] Churn distribution: {dict(df['Churn'].value_counts())}")

# ── 3. Feature Engineering ────────────────────────────────
df["ChargePerTenure"] = np.where(
    df["tenure"] > 0,
    df["TotalCharges"] / df["tenure"],
    df["MonthlyCharges"]
)
df["LongTermContract"] = df["Contract"].apply(
    lambda x: 1 if x in ["One year", "Two year"] else 0
)

# ── 4. Encode ─────────────────────────────────────────────
encoders = {}
cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, "encoders.pkl")
print(f"[INFO] Encoded {len(encoders)} categorical columns.")

# ── 5. Split ──────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

# ── 6. Define Models ──────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=4,
        subsample=0.8, random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=42, solver="saga"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=42
    ),
}

# ── 7. Train & Evaluate ───────────────────────────────────
print(f"\n{'Model':<25} {'CV AUC':>10} {'CV F1':>10} {'Test AUC':>10} {'Test F1':>10}")
print("-" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
best_model_name = None
best_auc = 0.0

for name, model in models.items():
    try:
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        cv_f1  = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1",      n_jobs=-1).mean()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)
        test_f1  = f1_score(y_test, y_pred)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"{name:<25} {cv_auc:>10.4f} {cv_f1:>10.4f} {test_auc:>10.4f} {test_f1:>10.4f}")
        results[name] = dict(
            model=model, cv_auc=cv_auc, cv_f1=cv_f1,
            test_auc=test_auc, test_f1=test_f1, test_acc=test_acc,
            y_pred=y_pred, y_prob=y_prob,
        )
        if test_auc > best_auc:
            best_auc = test_auc
            best_model_name = name
    except Exception as e:
        print(f"{name:<25} FAILED: {e}")

# ── 8. Report ─────────────────────────────────────────────
print(f"\n[BEST] {best_model_name} (AUC = {best_auc:.4f})")
best = results[best_model_name]
best_model = best["model"]
print("\nClassification Report:")
print(classification_report(y_test, best["y_pred"], target_names=["Stay", "Churn"]))
cm = confusion_matrix(y_test, best["y_pred"])
print(f"Confusion Matrix:\n  TN={cm[0][0]}  FP={cm[0][1]}\n  FN={cm[1][0]}  TP={cm[1][1]}")

# ── 9. Feature Importance ─────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    joblib.dump(importance, "feature_importance.pkl")
elif hasattr(best_model, "coef_"):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": np.abs(best_model.coef_[0])
    }).sort_values("Importance", ascending=False)
    joblib.dump(importance, "feature_importance.pkl")

# ── 10. LIME Explainability ───────────────────────────────
print("\n[INFO] Setting up LIME explainer...")
try:
    import lime
    import lime.lime_tabular

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X.columns),
        class_names=["Stay", "Churn"],
        mode="classification",
        random_state=42
    )
    joblib.dump(lime_explainer, "lime_explainer.pkl")
    print("[INFO] LIME saved to lime_explainer.pkl")

except ImportError:
    print("[WARN] lime not installed. Run: python -m pip install lime")
except Exception as e:
    print(f"[WARN] LIME skipped: {e}")

# ── 11. Save All Artifacts ────────────────────────────────
joblib.dump(best_model, "model.pkl")
joblib.dump(list(X.columns), "features.pkl")

metadata = {
    "model_name": best_model_name,
    "trained_at": datetime.now().isoformat(),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "test_accuracy": round(float(best["test_acc"]), 4),
    "test_auc":      round(float(best["test_auc"]), 4),
    "test_f1":       round(float(best["test_f1"]),  4),
    "all_model_results": {
        k: {
            "cv_auc":   round(float(v["cv_auc"]), 4),
            "test_auc": round(float(v["test_auc"]), 4),
            "test_f1":  round(float(v["test_f1"]),  4),
            "test_acc": round(float(v["test_acc"]), 4),
        }
        for k, v in results.items()
    },
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n[DONE] Saved: model.pkl | encoders.pkl | features.pkl")
print(f"             feature_importance.pkl | lime_explainer.pkl | model_metadata.json")
print(BANNER)
