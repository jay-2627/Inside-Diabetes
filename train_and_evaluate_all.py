"""
Inside Diabetes — Dual Model Training, Cross-Validation, Held-Out Test Evaluation,
XAI Computation, and Web Bundle Export Pipeline.

Models Preserved:
1. Artificial Neural Network (MLP: 8 -> 32 -> 16 -> 1, Sigmoid)
2. Support Vector Machine (SVM with RBF Kernel & Platt Probability Calibration)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)

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
    with open(DATA_PATH, 'r') as f:
        first_line = f.readline().strip()
    has_header = first_line.startswith('Pregnancies')
    
    df = pd.read_csv(DATA_PATH, header=0 if has_header else None)
    if not has_header or df.shape[1] == 9:
        df.columns = FEATURE_NAMES + ['Outcome']

    print(f"Dataset Loaded Successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Target Distribution: {dict(df['Outcome'].value_counts())}")
    return df

def run_experiment():
    df = load_and_audit_dataset()
    
    # ── PHASE 2: FAIR TRAIN/TEST SPLIT & DATA LEAKAGE PREVENTION ─────────────
    print("\n" + "=" * 70)
    print("PHASE 2: FAIR TRAIN/TEST SPLIT & DATA LEAKAGE PREVENTION")
    print("=" * 70)
    
    cols_zero_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_processed = df.copy()
    for col in cols_zero_missing:
        df_processed[col] = df_processed[col].replace(0, np.nan)
        
    X_raw = df_processed[FEATURE_NAMES]
    y = df_processed['Outcome'].values
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Training Samples: {len(X_train_raw)} (Positive: {sum(y_train == 1)}, Negative: {sum(y_train == 0)})")
    print(f"Held-Out Test Samples: {len(X_test_raw)} (Positive: {sum(y_test == 1)}, Negative: {sum(y_test == 0)})")
    
    train_medians = X_train_raw.median().to_dict()
    X_train_imp = X_train_raw.fillna(train_medians).values
    X_test_imp = X_test_raw.fillna(train_medians).values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # ── 5-Fold Stratified Cross Validation on Training Data ───────────────────
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING & 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # 1. Artificial Neural Network (ANN: MLP 8 -> 32 -> 16 -> 1)
    print("\n[1/2] Tuning & Cross-Validating Artificial Neural Network (ANN)...", flush=True)
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
        'hidden_layer_sizes': [(32, 16)],
        'alpha': [0.001, 0.01],
        'learning_rate_init': [0.01]
    }
    mlp_grid = GridSearchCV(mlp_base, mlp_param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
    mlp_grid.fit(X_train_scaled, y_train)
    best_mlp = mlp_grid.best_estimator_
    print(f"  Best ANN Params: {mlp_grid.best_params_}", flush=True)
    print(f"  Best 5-Fold Mean CV ROC-AUC: {mlp_grid.best_score_:.4f}", flush=True)
    
    # 2. Support Vector Machine (SVM with RBF Kernel & Platt Scaling)
    print("\n[2/2] Tuning & Cross-Validating Support Vector Machine (SVM)...", flush=True)
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
    
    models = {
        'Artificial Neural Network': (best_mlp, X_train_scaled, X_test_scaled),
        'Support Vector Machine': (best_svm, X_train_scaled, X_test_scaled)
    }
    
    # Find optimal threshold using training cross-validation
    print("\n" + "=" * 70, flush=True)
    print("THRESHOLD OPTIMIZATION ON TRAINING VALIDATION FOLDS", flush=True)
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
        
        f1_opt = f1_score(y_test, preds_opt)
        rec_opt = recall_score(y_test, preds_opt)
        prec_opt = precision_score(y_test, preds_opt)
        
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
        print(f"  Brier Score:   {brier:.4f}")
        
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(REPORTS_DIR, 'model_evaluation_metrics.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved held-out evaluation metrics to: {csv_path}")

    # ── PHASE 4: SAVING MODEL ARTIFACTS ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: SAVING ARTIFACTS & WEB DEPLOYMENT BUNDLE")
    print("=" * 70)
    
    joblib.dump(best_svm, os.path.join(MODEL_DIR, 'svm_model.pkl'))
    joblib.dump(best_mlp, os.path.join(MODEL_DIR, 'diabetes_ann.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    with open(os.path.join(MODEL_DIR, 'medians.json'), 'w') as f:
        json.dump(train_medians, f, indent=2)
        
    print("Saved active Python models (diabetes_ann.pkl, svm_model.pkl).")
    
    # Clean up deprecated model files if present
    for old_file in ['xgboost_model.pkl', 'rf_model.pkl', 'gb_model.pkl', 'gradient_boosting_calibrated.json']:
        old_path = os.path.join(MODEL_DIR, old_file)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
                print(f"Removed deprecated model artifact: {old_file}")
            except Exception as e:
                print(f"Could not remove {old_file}: {e}")

    # Extract ANN Weights & Biases
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
    
    # Global SVM sensitivity calculation
    global_importances_svm = {}
    base_eval_svm = best_svm.predict_proba(X_eval)[:, 1]
    for i, col in enumerate(FEATURE_NAMES):
        X_plus = X_eval.copy()
        X_plus[:, i] += eps
        probs_plus = best_svm.predict_proba(X_plus)[:, 1]
        grad = np.mean(np.abs((probs_plus - base_eval_svm) / eps))
        global_importances_svm[col] = float(grad)
    tot_imp_svm = sum(global_importances_svm.values())
    svm_shap_pct = {k: round((v / tot_imp_svm) * 100, 1) for k, v in global_importances_svm.items()}
    
    ann_payload = {
        'model_name': 'Explainable Feedforward ANN (MLP)',
        'architecture': f"8 Inputs -> {best_mlp.hidden_layer_sizes[0]} Hidden -> {best_mlp.hidden_layer_sizes[1]} Hidden -> 1 Output (Sigmoid)",
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

    # SVM Support Vectors & Dual Coefficients for Web Inference (RBF kernel)
    svm_sv = best_svm.support_vectors_.tolist()
    svm_dual_coef = best_svm.dual_coef_[0].tolist()
    svm_intercept = float(best_svm.intercept_[0])
    svm_gamma = float(best_svm._gamma)
    platt_a = round(-float(best_svm.probA_[0]), 8)
    platt_b = round(float(best_svm.probB_[0]), 8)

    # 5. Generate Dual-Model Client Bundle (ANN + SVM Only)
    web_models_bundle = {
        'models': [
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
                'dual_coef': svm_dual_coef,
                'shap_importance': svm_shap_pct
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
    
    js_content = "/* Inside Diabetes — Multi-Model Client-Side Engine (Self-Contained: ANN & SVM) */\n"
    js_content += f"window.ALL_MODELS_DATA = {json.dumps(web_models_bundle, indent=2)};\n"
    
    bundle_js_path = os.path.join(WEBSITE_DIR, 'models_bundle.js')
    with open(bundle_js_path, 'w', encoding='utf-8') as f:
        f.write(js_content)
    print(f"Generated offline-capable JS model bundle (ANN + SVM): {bundle_js_path}")

    return df_results, optimal_thresholds

if __name__ == '__main__':
    run_experiment()
