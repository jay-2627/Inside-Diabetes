# -*- coding: utf-8 -*-
"""
Inside Diabetes — Neural Network Training & XAI Pipeline
Model: Multilayer Perceptron (Artificial Neural Network: 8 -> 16 -> 8 -> 1)
Dataset: Pima Indians Diabetes Database (NIDDK)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
WEBSITE_DIR = os.path.join(BASE_DIR, 'website')

os.makedirs(MODEL_DIR, exist_ok=True)
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

def train_and_export():
    print(f"Loading dataset from: {DATA_PATH}")
    # Canonical Pima dataset has 8 features + 1 target (Outcome)
    df = pd.read_csv(DATA_PATH, header=None if not open(DATA_PATH).readline().startswith('Pregnancies') else 0)
    if df.shape[1] == 9:
        df.columns = FEATURE_NAMES + ['Outcome']
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {df['Outcome'].value_counts().to_dict()}")

    # 1. Medically responsible missing data handling:
    # 0 values in Glucose, BloodPressure, SkinThickness, Insulin, BMI are physiologically missing
    cols_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero_as_missing:
        df[col] = df[col].replace(0, np.nan)

    X = df[FEATURE_NAMES]
    y = df['Outcome']

    # 2. Stratified train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Compute training medians for imputation (prevent data leakage)
    medians = X_train.median().to_dict()
    print("Training medians for imputation:", medians)

    X_train_imputed = X_train.fillna(medians)
    X_test_imputed  = X_test.fillna(medians)

    # 3. Standardization (Z-score scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled  = scaler.transform(X_test_imputed)

    # 4. Train Feedforward Artificial Neural Network (MLP)
    # Architecture: 8 Inputs -> 16 Neurons (ReLU) -> 8 Neurons (ReLU) -> 1 Output (Sigmoid probability)
    print("Training Artificial Neural Network (MLPClassifier 8 -> 16 -> 8 -> 1)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.01,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25
    )
    mlp.fit(X_train_scaled, y_train)

    # 5. Evaluate genuine test performance metrics
    y_pred = mlp.predict(X_test_scaled)
    y_prob = mlp.predict_proba(X_test_scaled)[:, 1]

    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec  = float(recall_score(y_test, y_pred))
    f1   = float(f1_score(y_test, y_pred))
    auc  = float(roc_auc_score(y_test, y_prob))
    cm   = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- Model Evaluation Results (Test Set, N=154) ---")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # 6. Extract exact weight matrices and biases
    # mlp.coefs_: [W1 (8,16), W2 (16,8), W3 (8,1)]
    # mlp.intercepts_: [b1 (16), b2 (8), b3 (1)]
    weights = [w.tolist() for w in mlp.coefs_]
    biases  = [b.tolist() for b in mlp.intercepts_]

    # Baseline expected value on training population
    train_probs = mlp.predict_proba(X_train_scaled)[:, 1]
    expected_val = float(np.mean(train_probs))
    print(f"Baseline Expected Risk E[f(x)]: {expected_val:.4f} ({expected_val*100:.1f}%)")

    # Global feature sensitivity / gradient approximation
    # Calculate mean absolute gradient with respect to inputs across training set
    # Using finite differences on scaled features
    eps = 1e-4
    global_importances = {}
    X_eval = X_train_scaled[:200]
    base_eval_probs = mlp.predict_proba(X_eval)[:, 1]
    for i, col in enumerate(FEATURE_NAMES):
        X_plus = X_eval.copy()
        X_plus[:, i] += eps
        probs_plus = mlp.predict_proba(X_plus)[:, 1]
        grad = np.mean(np.abs((probs_plus - base_eval_probs) / eps))
        global_importances[col] = float(grad)
    
    # Normalize global importances to sum to 100%
    tot_imp = sum(global_importances.values())
    global_importance_pct = {k: round((v / tot_imp) * 100, 1) for k, v in global_importances.items()}
    print("\nGlobal Feature Importance Ranking:")
    for k, v in sorted(global_importance_pct.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:25s}: {v}%")

    # 7. Package complete payload
    payload = {
        'model_name': 'Explainable Feedforward ANN (MLP)',
        'architecture': '8 Inputs -> 16 Hidden (ReLU) -> 8 Hidden (ReLU) -> 1 Output (Sigmoid)',
        'dataset': 'Pima Indians Diabetes Database (NIDDK)',
        'total_samples': len(df),
        'metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': cm
        },
        'feature_names': FEATURE_NAMES,
        'feature_specs': FEATURE_SPECS,
        'medians': medians,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
        'weights': weights,
        'biases': biases,
        'expected_value': expected_val,
        'global_importance_pct': global_importance_pct
    }

    # Save to model/ and website/
    out_model_json = os.path.join(MODEL_DIR, 'model_weights.json')
    out_web_json   = os.path.join(WEBSITE_DIR, 'model_weights.json')
    
    with open(out_model_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    with open(out_web_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    # Save pickle artifacts
    joblib.dump(mlp, os.path.join(MODEL_DIR, 'diabetes_ann.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    print(f"\nSaved model weights to:\n  - {out_model_json}\n  - {out_web_json}")
    print("Training and export complete!")

if __name__ == '__main__':
    train_and_export()
