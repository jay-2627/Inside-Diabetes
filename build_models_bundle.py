"""
Build a single JavaScript bundle containing all model data for browser-side
multi-model comparison. This avoids fetch() calls (which fail on file://).
"""
import json, joblib, os, numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ── Paths ──────────────────────────────────────────────────────────────
DATA_CSV   = os.path.join('data', 'diabetes.csv')
MODEL_DIR  = 'model'
OUT_JS     = os.path.join('website', 'models_bundle.js')

# ── Load dataset & scaler ─────────────────────────────────────────────
import pandas as pd
df = pd.read_csv(DATA_CSV)
feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age']
X = df[feature_cols].values.copy()
y = df['Outcome'].values

# Impute zeros with column medians (same logic as training)
for col_i in range(X.shape[1]):
    col = X[:, col_i]
    if feature_cols[col_i] not in ['Pregnancies', 'Outcome']:
        mask = col == 0
        if mask.any():
            col[mask] = np.median(col[~mask])

scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
X_scaled = scaler.transform(X)

# Use same 80/20 split as training (random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# Raw (unscaled) test set for browser-side inference
X_raw_train, X_raw_test, _, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mean = scaler.mean_.tolist()
std  = scaler.scale_.tolist()

# ── Helper: compact tree export (limit depth) ─────────────────────────
def tree_to_dict(tree, max_depth=8, _depth=0):
    """Recursively serialize a sklearn tree, capping at max_depth."""
    def recurse(node, depth):
        if tree.children_left[node] == -1 or depth >= max_depth:
            # Leaf: for classifiers tree.value[node] has shape (1, n_classes)
            vals = tree.value[node].flatten()
            if len(vals) == 2:
                prob = vals[1] / vals.sum() if vals.sum() > 0 else 0.5
            else:
                prob = float(vals.mean())
            return {"v": round(prob, 5)}
        return {
            "f": int(tree.feature[node]),
            "t": round(float(tree.threshold[node]), 6),
            "l": recurse(tree.children_left[node], depth + 1),
            "r": recurse(tree.children_right[node], depth + 1)
        }
    return recurse(0, 0)

# ── Evaluate each model ───────────────────────────────────────────────
models_info = []

# 1. ANN (already embedded via model_weights.json – skip tree export,
#    but record accuracy for comparison)
try:
    ann = joblib.load(os.path.join(MODEL_DIR, 'diabetes_ann.pkl'))
    ann_probs = ann.predict_proba(X_test)[:, 1]
    ann_preds = (ann_probs >= 0.5).astype(int)
    models_info.append({
        "name": "ANN (Primary)",
        "type": "ann",
        "accuracy": round(float(accuracy_score(y_test, ann_preds)), 4),
        "auc":      round(float(roc_auc_score(y_test, ann_probs)), 4),
        "f1":       round(float(f1_score(y_test, ann_preds)), 4),
    })
    print(f"ANN: acc={models_info[-1]['accuracy']}, auc={models_info[-1]['auc']}")
except Exception as e:
    print(f"ANN load failed: {e}")

# 2. Random Forest
try:
    rf = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)
    # Export first 20 trees (compact)
    max_trees = min(20, len(rf.estimators_))
    trees = [tree_to_dict(est.tree_, max_depth=6) for est in rf.estimators_[:max_trees]]
    models_info.append({
        "name": "Random Forest",
        "type": "tree",
        "accuracy": round(float(accuracy_score(y_test, rf_preds)), 4),
        "auc":      round(float(roc_auc_score(y_test, rf_probs)), 4),
        "f1":       round(float(f1_score(y_test, rf_preds)), 4),
        "scaler_mean": mean,
        "scaler_std":  std,
        "trees": trees
    })
    print(f"RF:  acc={models_info[-1]['accuracy']}, auc={models_info[-1]['auc']}, trees={max_trees}")
except Exception as e:
    print(f"RF load failed: {e}")

# 3. Gradient Boosting
try:
    gb = joblib.load(os.path.join(MODEL_DIR, 'gb_model.pkl'))
    gb_probs = gb.predict_proba(X_test)[:, 1]
    gb_preds = (gb_probs >= 0.5).astype(int)
    # GB stores estimators_ as a 2D array: [n_estimators, 1]
    n_est = min(20, gb.n_estimators_)
    gb_trees = [tree_to_dict(gb.estimators_[i, 0].tree_, max_depth=5)
                for i in range(n_est)]
    models_info.append({
        "name": "Gradient Boosting",
        "type": "gb",
        "accuracy":      round(float(accuracy_score(y_test, gb_preds)), 4),
        "auc":           round(float(roc_auc_score(y_test, gb_probs)), 4),
        "f1":            round(float(f1_score(y_test, gb_preds)), 4),
        "scaler_mean":   mean,
        "scaler_std":    std,
        "learning_rate": round(float(gb.learning_rate), 4),
        "init_raw":      round(float(gb.init_.class_prior_[1]), 6),
        "trees":         gb_trees
    })
    print(f"GB:  acc={models_info[-1]['accuracy']}, auc={models_info[-1]['auc']}, trees={n_est}")
except Exception as e:
    print(f"GB load failed: {e}")

# 4. SVM (Calibrated)
try:
    svm = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
    svm_probs = svm.predict_proba(X_test)[:, 1]
    svm_preds = (svm_probs >= 0.5).astype(int)
    # Try to extract linear coefficients
    coefs, intercept = [], []
    try:
        base = svm.calibrated_classifiers_[0].estimator
        if hasattr(base, 'coef_'):
            coefs = base.coef_[0].tolist()
            intercept = base.intercept_.tolist()
    except:
        pass
    models_info.append({
        "name": "SVM (Calibrated)",
        "type": "svm",
        "accuracy":    round(float(accuracy_score(y_test, svm_preds)), 4),
        "auc":         round(float(roc_auc_score(y_test, svm_probs)), 4),
        "f1":          round(float(f1_score(y_test, svm_preds)), 4),
        "scaler_mean": mean,
        "scaler_std":  std,
        "coefficients": [round(c, 6) for c in coefs],
        "intercept":    [round(c, 6) for c in intercept],
    })
    print(f"SVM: acc={models_info[-1]['accuracy']}, auc={models_info[-1]['auc']}")
except Exception as e:
    print(f"SVM load failed: {e}")

# ── Write JS bundle ───────────────────────────────────────────────────
js_data = json.dumps(models_info, separators=(',', ':'))
js_content = f"// Auto-generated model bundle — do not edit manually\nconst COMPARISON_MODELS = {js_data};\n"

with open(OUT_JS, 'w') as f:
    f.write(js_content)

size_kb = os.path.getsize(OUT_JS) / 1024
print(f"\nWrote {OUT_JS} ({size_kb:.1f} KB) with {len(models_info)} models")
