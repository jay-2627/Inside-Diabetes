# Artificial Neural Network Architecture & Clinical Validation Report
## Inside Diabetes: Explainable Deep MLP Benchmarking & Deployment

**Project:** Inside Diabetes — Clinical Risk Prediction Platform  
**Dataset:** Pima Indians Diabetes Database (NIDDK)  
**Evaluator:** Machine Learning & Clinical Systems Team  
**Model:** Artificial Neural Network (Feedforward Multi-Layer Perceptron: $8 \to 32 \to 16 \to 1$)  
**Status:** Validated, Reproducible, and Deployed to Live Platform  

---

## Executive Summary

The clinical risk engine for the *Inside Diabetes* platform is powered by an optimized **Feedforward Artificial Neural Network (ANN - Deep MLP)**:
- **Topology:** $8\text{ Inputs} \longrightarrow 32\text{ Hidden (ReLU)} \longrightarrow 16\text{ Hidden (ReLU)} \longrightarrow 1\text{ Output (Sigmoid)}$
- **Training Strategy:** 5-Fold Stratified Cross-Validation on $N=614$ training samples with L2 weight decay ($\alpha=0.01$) and validation early stopping.
- **Evaluation Set:** Strictly partitioned, held-out test cohort ($N=154$, 100 non-diabetic, 54 diabetic).

### Held-Out Performance Metrics:
- **Accuracy:** `73.38%`
- **ROC-AUC:** `0.8191` (95% Bootstrap CI: `[0.7434, 0.8820]`)
- **Precision:** `64.44%`
- **Specificity:** `84.00%`
- **Brier Score:** `0.1668`
- **Default Recall (0.50 Threshold):** `53.70%` (29/54 detected)
- **Clinical Screening Recall (0.25 Threshold):** `85.19%` (46/54 detected; only 8 false negatives)
- **F1-Score at 0.25 Threshold:** `0.6667`

---

## Client-Side In-Browser Inference & Explainability

1. **Exact Mathematical Parity:** Layer weights and biases are exported to `website/models_bundle.js`. The client-side JavaScript engine replicates the forward pass with zero discrepancy ($\Delta = 0.00000000$ across 20 distinct test cases).
2. **Integrated Gradients XAI:** Individual prediction contributions are calculated along linear interpolation paths between cohort baselines and patient values.
3. **Counterfactual Exploration:** Actionable lifestyle recourse solver interactively calculates biomarker adjustments needed to reduce predicted risk below $30\%$, while strictly locking immutable features (`Age`, `Pregnancies`, and `Pedigree`).
