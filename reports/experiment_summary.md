# Inside Diabetes: Empirical Validation & Neural Network Analysis

**Senior Machine Learning & Clinical Decision Review**  
**Dataset:** Pima Indians Diabetes Database ($N=768$, 8 clinical biomarkers + 1 binary target)  
**Evaluation Cohort:** Strictly partitioned held-out test split ($N=154$, 20% stratified: 100 negatives, 54 positives; $N_{\text{train}}=614$, 80% stratified; `random_state=42`)

---

## 1. Executive Summary

A rigorous empirical evaluation of the **Artificial Neural Network (Deep MLP: $8 \to 32 \to 16 \to 1$)** was conducted. All imputers and scalers were fitted exclusively on training data (within cross-validation folds) to guarantee zero data leakage.

### Held-Out Performance Metrics:
The saved production model reproduces the metrics in [reports/model_evaluation_metrics.csv](file:///c:/ANN/Diabetes_AI_Project/reports/model_evaluation_metrics.csv):
- **Accuracy:** `73.38%` (113/154 correct: TN=84, TP=29)
- **ROC-AUC:** `0.8191` (95% Bootstrap CI: `[0.7434, 0.8820]`)
- **Precision:** `64.44%`
- **Specificity:** `84.00%`
- **Brier Score:** `0.1668` (Well-calibrated probability)
- **Expected Calibration Error (ECE):** `0.0921`
- **Recall at Default Threshold (0.50):** `53.70%` (29/54 detected)
- **Recall at Validation-Optimal Threshold (0.25):** `85.19%` (46/54 detected; catches 85.2% of diabetic patients)

---

## 2. Key Methodological Validations

1. **Zero Data Leakage:** Physiological zeros across `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were converted to `NaN` and imputed using **training-set median values** exclusively.
2. **Clinical Screening Optimization:** Operating at the validation-derived threshold of $\theta^* = 0.25$ slashes False Negatives from 25 down to just 8, significantly reducing clinical risk.
3. **Exact Mathematical Parity:** Client-side JavaScript (`website/models_bundle.js`) replicates the feedforward matrix multiplications with a numerical discrepancy of $0.00000000$ across 20 distinct clinical profiles.
4. **Explainable AI:** Local feature attributions are computed via **Integrated Gradients** over the continuous ReLU manifolds, identifying Glucose (~33%) and BMI (~24%) as the primary metabolic drivers.
