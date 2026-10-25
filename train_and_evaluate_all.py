"""
Inside Diabetes — Comprehensive Four-Model Training, Hyperparameter Tuning,
Cross-Validation, Held-Out Test Evaluation, SHAP XAI, and Artifact Export Pipeline.

Models:
1. XGBoost Classifier
2. Random Forest Classifier
3. Support Vector Machine (SVM, Calibrated)
4. Artificial Neural Network (MLPClassifier: 8 -> 16 -> 8 -> 1)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, log_loss
)
import xgboost as xgb
import shap

# ── 1. Directories & Paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
WEBSITE_DIR = os.path.join(BASE_DIR, 'website')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(WEBSITE_DIR, exist_ok=True)

FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]
FEATURE_SPECS = {
    'Pregnancies': {'unit': 'count', 'min': 0, 'max': 20, 'default': 1, 'step': 1},
    'Glucose': {'unit': 'mg/dL', 'min': 40, 'max': 300, 'default': 100, 'step': 1},
    'BloodPressure': {'unit': 'mm Hg', 'min': 30, 'max': 180, 'default': 72, 'step': 1},
    'SkinThickness': {'unit': 'mm', 'min': 5, 'max': 99, 'default': 23, 'step': 1},
    'Insulin': {'unit': 'µU/mL', 'min': 10, 'max': 850, 'default': 85, 'step': 1},
    'BMI': {'unit': 'kg/m²', 'min': 12.0, 'max': 67.0, 'default': 26.5, 'step': 0.1},
    'DiabetesPedigreeFunction': {'unit': 'score', 'min': 0.05, 'max': 2.50, 'default': 0.38, 'step': 0.01},
    'Age': {'unit': 'years', 'min': 18, 'max': 100, 'default': 33, 'step': 1}
}

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_and_audit_dataset():
    print("=" * 70)
    print("PHASE 1: DATASET INGESTION & AUDIT")
    print("=" * 70)
    # Check if header is present
    with open(DATA_PATH, 'r') as f:
        first_line = f.readline().strip()
    has_header = first_line.startswith('Pregnancies')
    
    df = pd.read_csv(DATA_PATH, header=0 if has_header else None)
    if not has_header or df.shape[1] == 9:
        df.columns = FEATURE_NAMES + ['Outcome']

    print(f"Dataset Loaded Successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Duplicates: {df.duplicated().sum()}")
    print(f"Target Distribution: {dict(df['Outcome'].value_counts())}")
    print(f"Negative (0): {df['Outcome'].value_counts()[0]} ({df['Outcome'].value_counts(normalize=True)[0]*100:.2f}%)")
    print(f"Positive (1): {df['Outcome'].value_counts()[1]} ({df['Outcome'].value_counts(normalize=True)[1]*100:.2f}%)")
    
    return df

def run_experiment():
    df = load_and_audit_dataset()
    
    # ── PHASE 2: FAIR MODEL TRAINING & DATA PARTITIONING ─────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: FAIR TRAIN/TEST SPLIT & DATA LEAKAGE PREVENTION")
    print("=" * 70)
    
    # Mark physiologically impossible zeros as NaN
    cols_zero_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_processed = df.copy()
    for col in cols_zero_missing:
        df_processed[col] = df_processed[col].replace(0, np.nan)
        
    X_raw = df_processed[FEATURE_NAMES]
    y = df_processed['Outcome'].values
    
    # Stratified 80/20 train/test split. Test set is strictly held-out.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Training Samples: {len(X_train_raw)} (Positive: {sum(y_train == 1)}, Negative: {sum(y_train == 0)})")
    print(f"Held-Out Test Samples: {len(X_test_raw)} (Positive: {sum(y_test == 1)}, Negative: {sum(y_test == 0)})")
    
    # Training set medians (Strictly computed on X_train to prevent leakage)
    train_medians = X_train_raw.median().to_dict()
    print("\nTraining-fold Medians for Imputation (No test data seen):")
    for k, v in train_medians.items():
        print(f"  {k:25s}: {v:.2f}")
        
    # Impute missing values with training medians
    X_train_imp = X_train_raw.fillna(train_medians).values
    X_test_imp = X_test_raw.fillna(train_medians).values
    
    # Scaler fitted strictly on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # ── 5-Fold Stratified Cross Validation on Training Data ───────────────────
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING & 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # 1. XGBoost
    print("\n[1/4] Tuning & Cross-Validating XGBoost...")
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_base = xgb.XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric='logloss',
        scale_pos_weight=scale_pos
    )
    xgb_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 4],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.9],
        'colsample_bytree': [0.9]
    }
    xgb_grid = GridSearchCV(xgb_base, xgb_param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
    xgb_grid.fit(X_train_imp, y_train)
    best_xgb = xgb_grid.best_estimator_
    print(f"  Best XGB Params: {xgb_grid.best_params_}", flush=True)
    print(f"  Best 5-Fold Mean CV ROC-AUC: {xgb_grid.best_score_:.4f}", flush=True)
    
    # 2. Random Forest
    print("\n[2/4] Tuning & Cross-Validating Random Forest...", flush=True)
    rf_base = RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced')
    rf_param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [4, 6],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
    rf_grid.fit(X_train_imp, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"  Best RF Params: {rf_grid.best_params_}", flush=True)
    print(f"  Best 5-Fold Mean CV ROC-AUC: {rf_grid.best_score_:.4f}", flush=True)
    
    # 3. Support Vector Machine (with probability calibration via Platt scaling)
    print("\n[3/4] Tuning & Cross-Validating Support Vector Machine (SVM)...", flush=True)
    svm_base = SVC(probability=True, random_state=RANDOM_SEED, class_weight='balanced')
    svm_param_grid = {
        'C': [0.5, 1.0, 5.0],
        'gamma': ['scale', 0.05],
        'kernel': ['rbf']
    }
    svm_grid = GridSearchCV(svm_base, svm_param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
    svm_grid.fit(X_train_scaled, y_train)
    best_svm = svm_grid.best_estimator_
    print(f"  Best SVM Params: {svm_grid.best_params_}", flush=True)
    print(f"  Best 5-Fold Mean CV ROC-AUC: {svm_grid.best_score_:.4f}", flush=True)
    
    # 4. Artificial Neural Network (MLP 8 -> 16 -> 8 -> 1)
    print("\n[4/4] Tuning & Cross-Validating Artificial Neural Network (ANN)...", flush=True)
    mlp_base = MLPClassifier(
        random_state=RANDOM_SEED,
        max_iter=1000,
        activation='relu',
        solver='adam',
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25
    )
    mlp_param_grid = {
        'hidden_layer_sizes': [(16, 8), (32, 16)],
        'alpha': [0.001, 0.01],
        'learning_rate_init': [0.01]
    }
    mlp_grid = GridSearchCV(mlp_base, mlp_param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
    mlp_grid.fit(X_train_scaled, y_train)
    best_mlp = mlp_grid.best_estimator_
    print(f"  Best ANN Params: {mlp_grid.best_params_}", flush=True)
    print(f"  Best 5-Fold Mean CV ROC-AUC: {mlp_grid.best_score_:.4f}", flush=True)
    
    # Collect CV performance across all folds
    models = {
        'XGBoost': (best_xgb, X_train_imp, X_test_imp),
        'Random Forest': (best_rf, X_train_imp, X_test_imp),
        'Support Vector Machine': (best_svm, X_train_scaled, X_test_scaled),
        'Artificial Neural Network': (best_mlp, X_train_scaled, X_test_scaled)
    }
    
    # Find optimal threshold using training cross-validation (prevent test leakage)
    print("\n" + "=" * 70, flush=True)
    print("THRESHOLD OPTIMIZATION ON TRAINING VALIDATION FOLDS (F1-Optimal)", flush=True)
    print("=" * 70, flush=True)
    optimal_thresholds = {}
    from sklearn.base import clone
    for name, (model, X_tr, _) in models.items():
        oof_probs = np.zeros(len(y_train))
        for train_idx, val_idx in cv.split(X_tr, y_train):
            m_clone = clone(model)
            m_clone.fit(X_tr[train_idx], y_train[train_idx])
            oof_probs[val_idx] = m_clone.predict_proba(X_tr[val_idx])[:, 1]
            
        thresholds = np.linspace(0.2, 0.8, 61)
        f1_scores = [f1_score(y_train, oof_probs >= t) for t in thresholds]
        best_idx = np.argmax(f1_scores)
        opt_thresh = float(thresholds[best_idx])
        optimal_thresholds[name] = opt_thresh
        print(f"  {name:25s} | Optimal Val Threshold: {opt_thresh:.2f} (OOF F1: {f1_scores[best_idx]:.4f})", flush=True)

    # ── PHASE 3: EVALUATION ON HELD-OUT TEST SET ─────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: RIGOROUS EVALUATION ON HELD-OUT TEST SET (N=154)")
    print("=" * 70)
    
    results = []
    
    for name, (model, _, X_te) in models.items():
        probs = model.predict_proba(X_te)[:, 1]
        opt_thresh = optimal_thresholds[name]
        preds_default = (probs >= 0.5).astype(int)
        preds_opt = (probs >= opt_thresh).astype(int)
        
        # Primary evaluation using default threshold 0.50
        acc = accuracy_score(y_test, preds_default)
        prec = precision_score(y_test, preds_default)
        rec = recall_score(y_test, preds_default)
        f1 = f1_score(y_test, preds_default)
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        cm = confusion_matrix(y_test, preds_default)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        
        # Also at optimal threshold
        acc_opt = accuracy_score(y_test, preds_opt)
        rec_opt = recall_score(y_test, preds_opt)
        prec_opt = precision_score(y_test, preds_opt)
        f1_opt = f1_score(y_test, preds_opt)
        cm_opt = confusion_matrix(y_test, preds_opt)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall (Sensitivity)': rec,
            'Specificity': specificity,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'Brier Score': brier,
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp),
            'Optimal Threshold': opt_thresh,
            'F1 at Opt Threshold': f1_opt,
            'Recall at Opt Threshold': rec_opt,
            'Precision at Opt Threshold': prec_opt
        })
        
        print(f"\n--- {name} ---")
        print(f"  Accuracy:      {acc:.4f} ({acc*100:.1f}%)")
        print(f"  Precision:     {prec:.4f}")
        print(f"  Recall:        {rec:.4f}")
        print(f"  Specificity:   {specificity:.4f}")
        print(f"  F1-Score:      {f1:.4f}")
        print(f"  ROC-AUC:       {roc_auc:.4f}")
        print(f"  PR-AUC:        {pr_auc:.4f}")
        print(f"  Brier Score:   {brier:.4f} (lower is better)")
        print(f"  Confusion Matrix (Threshold=0.50):")
        print(f"    [[TN={tn:2d}, FP={fp:2d}],")
        print(f"     [FN={fn:2d}, TP={tp:2d}]]")
        print(f"  Performance at Validation-Tuned Threshold ({opt_thresh:.2f}):")
        print(f"    F1={f1_opt:.4f}, Recall={rec_opt:.4f}, Precision={prec_opt:.4f}")
        print(f"    Confusion Matrix: {cm_opt.tolist()}")
        
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(REPORTS_DIR, 'model_evaluation_metrics.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved held-out evaluation metrics to: {csv_path}")
    
    # ── PHASE 4: MODEL SELECTION ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: COMPARATIVE ANALYSIS & SELECTION")
    print("=" * 70)
    
    best_prec = df_results.loc[df_results['Precision'].idxmax()]
    best_rec = df_results.loc[df_results['Recall (Sensitivity)'].idxmax()]
    best_f1 = df_results.loc[df_results['F1-Score'].idxmax()]
    best_auc = df_results.loc[df_results['ROC-AUC'].idxmax()]
    best_calib = df_results.loc[df_results['Brier Score'].idxmin()]
    
    print(f"Highest Precision:           {best_prec['Model']} ({best_prec['Precision']:.4f})")
    print(f"Highest Recall:              {best_rec['Model']} ({best_rec['Recall (Sensitivity)']:.4f})")
    print(f"Highest F1-Score:            {best_f1['Model']} ({best_f1['F1-Score']:.4f})")
    print(f"Highest ROC-AUC:             {best_auc['Model']} ({best_auc['ROC-AUC']:.4f})")
    print(f"Best Calibration (Min Brier):{best_calib['Model']} ({best_calib['Brier Score']:.4f})")
    
    # ── PHASE 5: EXPLAINABLE AI (SHAP) ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 5: EXPLAINABLE AI (SHAP) COMPUTATION")
    print("=" * 70)
    
    # XGBoost TreeSHAP
    print("Computing TreeSHAP for XGBoost...")
    explainer_xgb = shap.TreeExplainer(best_xgb)
    shap_vals_xgb = explainer_xgb.shap_values(X_test_imp)
    xgb_mean_shap = np.mean(np.abs(shap_vals_xgb), axis=0)
    xgb_shap_pct = {FEATURE_NAMES[i]: round(float(xgb_mean_shap[i] / np.sum(xgb_mean_shap) * 100), 2) for i in range(8)}
    print("XGBoost SHAP Feature Importance (%):", xgb_shap_pct)
    
    # Random Forest TreeSHAP
    print("Computing TreeSHAP for Random Forest...")
    explainer_rf = shap.TreeExplainer(best_rf)
    shap_vals_rf = explainer_rf.shap_values(X_test_imp)
    # Binary classification: index 1 corresponds to positive class
    if isinstance(shap_vals_rf, list):
        rf_mean_shap = np.mean(np.abs(shap_vals_rf[1]), axis=0)
    elif shap_vals_rf.ndim == 3:
        rf_mean_shap = np.mean(np.abs(shap_vals_rf[:, :, 1]), axis=0)
    else:
        rf_mean_shap = np.mean(np.abs(shap_vals_rf), axis=0)
    rf_shap_pct = {FEATURE_NAMES[i]: round(float(rf_mean_shap[i] / np.sum(rf_mean_shap) * 100), 2) for i in range(8)}
    print("Random Forest SHAP Feature Importance (%):", rf_shap_pct)

    # ── PHASE 6: ARTIFACT EXPORT & BUNDLING ──────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 6: SAVING MODEL ARTIFACTS & WEB DEPLOYMENT BUNDLE")
    print("=" * 70)
    
    # 1. Save standard Python pickle artifacts
    joblib.dump(best_xgb, os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    joblib.dump(best_rf, os.path.join(MODEL_DIR, 'rf_model.pkl'))
    joblib.dump(best_svm, os.path.join(MODEL_DIR, 'svm_model.pkl'))
    joblib.dump(best_mlp, os.path.join(MODEL_DIR, 'diabetes_ann.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    # Save training medians
    with open(os.path.join(MODEL_DIR, 'medians.json'), 'w') as f:
        json.dump(train_medians, f, indent=2)
        
    print("Saved all Python models (.pkl) to 'model/' directory.")
    
    # 2. Extract ANN Weights & Biases for authentic JavaScript forward-pass & SHAP
    mlp_weights = [w.tolist() for w in best_mlp.coefs_]
    mlp_biases = [b.tolist() for b in best_mlp.intercepts_]
    train_probs_mlp = best_mlp.predict_proba(X_train_scaled)[:, 1]
    expected_val_mlp = float(np.mean(train_probs_mlp))
    
    # Global ANN sensitivity calculation
    eps = 1e-4
    global_importances_ann = {}
    X_eval = X_train_scaled[:200]
    base_eval_probs = best_mlp.predict_proba(X_eval)[:, 1]
    for i, col in enumerate(FEATURE_NAMES):
        X_plus = X_eval.copy()
        X_plus[:, i] += eps
        probs_plus = best_mlp.predict_proba(X_plus)[:, 1]
        grad = np.mean(np.abs((probs_plus - base_eval_probs) / eps))
        global_importances_ann[col] = float(grad)
    tot_imp = sum(global_importances_ann.values())
    ann_shap_pct = {k: round((v / tot_imp) * 100, 1) for k, v in global_importances_ann.items()}
    
    ann_payload = {
        'model_name': 'Explainable Feedforward ANN (MLP)',
        'architecture': f"8 Inputs -> {best_mlp.hidden_layer_sizes[0]} Hidden -> {best_mlp.hidden_layer_sizes[1] if len(best_mlp.hidden_layer_sizes)>1 else ''} Hidden -> 1 Output (Sigmoid)",
        'dataset': 'Pima Indians Diabetes Database (NIDDK)',
        'total_samples': len(df),
        'metrics': {
            'accuracy': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'Accuracy'].values[0]),
            'precision': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'Precision'].values[0]),
            'recall': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'Recall (Sensitivity)'].values[0]),
            'f1_score': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'F1-Score'].values[0]),
            'roc_auc': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'ROC-AUC'].values[0]),
            'confusion_matrix': confusion_matrix(y_test, (best_mlp.predict_proba(X_test_scaled)[:, 1] >= 0.5).astype(int)).tolist()
        },
        'feature_names': FEATURE_NAMES,
        'feature_specs': FEATURE_SPECS,
        'medians': train_medians,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
        'weights': mlp_weights,
        'biases': mlp_biases,
        'expected_value': expected_val_mlp,
        'global_importance_pct': ann_shap_pct
    }
    
    with open(os.path.join(MODEL_DIR, 'model_weights.json'), 'w') as f:
        json.dump(ann_payload, f, indent=2)
    with open(os.path.join(WEBSITE_DIR, 'model_weights.json'), 'w') as f:
        json.dump(ann_payload, f, indent=2)

    # 3. Create Compact JS Representation for XGBoost and Random Forest
    def sklearn_tree_to_dict(tree, max_depth=6):
        def recurse(node, depth):
            vals = tree.value[node].flatten()
            prob = vals[1] / vals.sum() if vals.sum() > 0 else 0.5
            if tree.children_left[node] == -1 or depth >= max_depth:
                return {"v": round(float(prob), 4)}
            return {
                "f": int(tree.feature[node]),
                "t": round(float(tree.threshold[node]), 4),
                "ev": round(float(prob), 4),
                "l": recurse(tree.children_left[node], depth + 1),
                "r": recurse(tree.children_right[node], depth + 1)
            }
        return recurse(0, 0)
        
    rf_trees = [sklearn_tree_to_dict(est.tree_, max_depth=6) for est in best_rf.estimators_]
    
    # Export XGBoost trees using Dump Model JSON with stats
    xgb_booster = best_xgb.get_booster()
    xgb_dump = xgb_booster.get_dump(dump_format='json', with_stats=True)
    
    def parse_xgb_node(node):
        if 'leaf' in node:
            return {'v': round(float(node['leaf']), 4)}, float(node.get('cover', 1.0))
        feat_name = node['split']
        feat_idx = int(feat_name.replace('f', '')) if feat_name.startswith('f') else FEATURE_NAMES.index(feat_name)
        thresh = float(node['split_condition'])
        yes_id = node['yes']
        no_id = node['no']
        
        children = {c['nodeid']: c for c in node['children']}
        l_dict, l_cov = parse_xgb_node(children[yes_id])
        r_dict, r_cov = parse_xgb_node(children[no_id])
        tot_cov = l_cov + r_cov
        l_val = l_dict.get('v', l_dict.get('ev', 0.0))
        r_val = r_dict.get('v', r_dict.get('ev', 0.0))
        ev_node = (l_val * l_cov + r_val * r_cov) / tot_cov if tot_cov > 0 else 0.0
        return {
            'f': feat_idx,
            't': round(thresh, 4),
            'ev': round(ev_node, 4),
            'l': l_dict,
            'r': r_dict
        }, tot_cov
        
    xgb_trees = []
    for raw_tree_str in xgb_dump:
        tree_json = json.loads(raw_tree_str)
        parsed_tree, _ = parse_xgb_node(tree_json)
        xgb_trees.append(parsed_tree)

    # SVM Support Vectors & Dual Coefficients for Web Inference (RBF kernel)
    svm_sv = best_svm.support_vectors_.tolist()
    svm_dual_coef = best_svm.dual_coef_[0].tolist()
    svm_intercept = float(best_svm.intercept_[0])
    svm_gamma = float(best_svm._gamma)
    # Platt calibration: prob = 1 / (1 + exp(-(platt_a * df + platt_b)))
    # Fitted from svm_model.pkl (probA_ and probB_)
    platt_a = round(-float(best_svm.probA_[0]), 8)
    platt_b = round(float(best_svm.probB_[0]), 8)

    # 4. Generate Single Self-Contained JavaScript Bundle (Zero CORS Dependency)
    web_models_bundle = {
        'models': [
            {
                'id': 'xgboost',
                'name': 'XGBoost Classifier',
                'badge': 'Gradient Boosted Trees',
                'accuracy': float(df_results.loc[df_results['Model']=='XGBoost', 'Accuracy'].values[0]),
                'roc_auc': float(df_results.loc[df_results['Model']=='XGBoost', 'ROC-AUC'].values[0]),
                'f1': float(df_results.loc[df_results['Model']=='XGBoost', 'F1-Score'].values[0]),
                'recall': float(df_results.loc[df_results['Model']=='XGBoost', 'Recall (Sensitivity)'].values[0]),
                'precision': float(df_results.loc[df_results['Model']=='XGBoost', 'Precision'].values[0]),
                'specificity': float(df_results.loc[df_results['Model']=='XGBoost', 'Specificity'].values[0]),
                'brier': float(df_results.loc[df_results['Model']=='XGBoost', 'Brier Score'].values[0]),
                'type': 'xgboost_tree',
                'trees': xgb_trees,
                'learning_rate': float(best_xgb.learning_rate),
                'shap_importance': xgb_shap_pct
            },
            {
                'id': 'rf',
                'name': 'Random Forest Classifier',
                'badge': 'Bagged Ensembles',
                'accuracy': float(df_results.loc[df_results['Model']=='Random Forest', 'Accuracy'].values[0]),
                'roc_auc': float(df_results.loc[df_results['Model']=='Random Forest', 'ROC-AUC'].values[0]),
                'f1': float(df_results.loc[df_results['Model']=='Random Forest', 'F1-Score'].values[0]),
                'recall': float(df_results.loc[df_results['Model']=='Random Forest', 'Recall (Sensitivity)'].values[0]),
                'precision': float(df_results.loc[df_results['Model']=='Random Forest', 'Precision'].values[0]),
                'specificity': float(df_results.loc[df_results['Model']=='Random Forest', 'Specificity'].values[0]),
                'brier': float(df_results.loc[df_results['Model']=='Random Forest', 'Brier Score'].values[0]),
                'type': 'rf_tree',
                'trees': rf_trees,
                'shap_importance': rf_shap_pct
            },
            {
                'id': 'svm',
                'name': 'Support Vector Machine (SVM)',
                'badge': 'RBF Kernel (Calibrated)',
                'accuracy': float(df_results.loc[df_results['Model']=='Support Vector Machine', 'Accuracy'].values[0]),
                'roc_auc': float(df_results.loc[df_results['Model']=='Support Vector Machine', 'ROC-AUC'].values[0]),
                'f1': float(df_results.loc[df_results['Model']=='Support Vector Machine', 'F1-Score'].values[0]),
                'recall': float(df_results.loc[df_results['Model']=='Support Vector Machine', 'Recall (Sensitivity)'].values[0]),
                'precision': float(df_results.loc[df_results['Model']=='Support Vector Machine', 'Precision'].values[0]),
                'specificity': float(df_results.loc[df_results['Model']=='Support Vector Machine', 'Specificity'].values[0]),
                'brier': float(df_results.loc[df_results['Model']=='Support Vector Machine', 'Brier Score'].values[0]),
                'type': 'svm_rbf',
                'gamma': svm_gamma,
                'intercept': svm_intercept,
                'platt_a': platt_a,
                'platt_b': platt_b,
                'support_vectors': svm_sv,
                'dual_coef': svm_dual_coef
            },
            {
                'id': 'ann',
                'name': 'Artificial Neural Network (ANN)',
                'badge': 'Deep MLP (8->32->16->1)',
                'accuracy': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'Accuracy'].values[0]),
                'roc_auc': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'ROC-AUC'].values[0]),
                'f1': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'F1-Score'].values[0]),
                'recall': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'Recall (Sensitivity)'].values[0]),
                'precision': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'Precision'].values[0]),
                'specificity': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'Specificity'].values[0]),
                'brier': float(df_results.loc[df_results['Model']=='Artificial Neural Network', 'Brier Score'].values[0]),
                'type': 'ann_mlp',
                'weights': mlp_weights,
                'biases': mlp_biases,
                'expected_value': expected_val_mlp,
                'shap_importance': ann_shap_pct
            }
        ],
        'metadata': {
            'feature_names': FEATURE_NAMES,
            'feature_specs': FEATURE_SPECS,
            'medians': train_medians,
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_std': scaler.scale_.tolist()
        }
    }
    
    js_content = "/* Inside Diabetes — Multi-Model Client-Side Engine (Self-Contained) */\n"
    js_content += f"window.ALL_MODELS_DATA = {json.dumps(web_models_bundle, indent=2)};\n"
    
    bundle_js_path = os.path.join(WEBSITE_DIR, 'models_bundle.js')
    with open(bundle_js_path, 'w', encoding='utf-8') as f:
        f.write(js_content)
    print(f"Generated offline-capable JS model bundle: {bundle_js_path}")

    return df_results, optimal_thresholds

if __name__ == '__main__':
    run_experiment()
