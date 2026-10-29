# Inside Diabetes — Cinematic Medical Documentary & Explainable ANN Risk Prediction

An immersive, scroll-driven 3D biological exploration and explainable artificial intelligence (XAI) risk prediction platform for diabetes powered by a Deep Multilayer Perceptron (ANN).

---

## 🌟 Key Highlights

- **3D Cinematic Biology:** Built with Three.js featuring continuous Steadicam camera physics, realistic anatomical models (Full Human Anatomy, Vascular Bloodstream with Erythrocytes & Glucose, Pancreatic Beta Cells, and Cellular Insulin Receptor Signaling).
- **Pure Artificial Neural Network (ANN):** Feedforward Deep MLP ($8 \to 32 \to 16 \to 1$, ReLU, Sigmoid output) trained on the authentic NIDDK Pima Indians Diabetes Database with zero data leakage.
- **Clinical Screening Optimization:** Operating at a validation-optimal threshold of $\theta^* = 0.25$, boosting diabetic detection sensitivity from $53.7\%$ to **$85.2\%$**.
- **Integrated Gradients XAI:** Computes authentic path integrals of neural activations for patient-specific feature attribution.
- **Counterfactual Actionable Recourse:** An iterative lifestyle scenario solver that interactively determines minimal biomarker modifications needed to achieve safe metabolic levels (< 30% risk).
- **Exact Python-JavaScript Parity:** Zero-latency client-side inference executing directly in the browser with verified mathematical parity ($\Delta = 0.00000000$).

---

## 📁 Project Directory Structure

```
Diabetes_AI_Project/
├── data/                         # Clinical Datasets
│   └── diabetes.csv              # Authentic NIDDK Pima Indians Diabetes Database
├── model/                        # Production Model Artifacts
│   ├── diabetes_ann.pkl          # Trained Deep MLP model artifact
│   ├── scaler.pkl                # Standardizer fitted strictly on training cohort
│   ├── medians.json              # Cohort imputation medians
│   └── model_weights.json        # Exported neural weights & biases
├── reports/                      # Clinical Evaluation & Validation Reports
│   ├── model_evaluation_metrics.csv # Evaluated test-set metrics
│   ├── bootstrap_auc_ci.csv      # Bootstrap 95% confidence intervals
│   ├── calibration_ece.csv       # Expected calibration error & Brier scores
│   └── experiment_summary.md     # Methodological & validation documentation
├── scripts/                      # Utility Scripts
│   ├── build_models_bundle.py    # Client bundle compilation helper
│   ├── clean_hologram.py         # Hologram alpha processing script
│   └── process_human_image.py    # Anatomical image preparation
├── tests/                        # Verification & Parity Test Suite
│   ├── test_model_verification.py# 5-test end-to-end verification suite
│   └── check_js_parity.py        # 20-case Python vs JS numerical parity test
├── website/                      # Client-Side Medical Web Application
│   ├── index.html                # Main documentary website & risk calculator
│   ├── style.css                 # Cinematic CSS design system
│   ├── script.js                 # Three.js 3D biology & ANN inference engine
│   ├── models_bundle.js          # Standalone client-side JS neural engine (33 KB)
│   ├── model_weights.json        # Configuration & model parameters
│   ├── assets/                   # Images, holograms, textures
│   └── lib/                      # Vendor libraries (Three.js, GSAP)
├── train_and_evaluate_all.py     # Reproducible Pipeline (Training, CV, Evaluation, Export)
├── ANN_MODEL_REPORT.md           # Formal evaluation and benchmarking report
├── RESEARCH_PROBLEM.md           # Research framework and clinical objectives
├── vercel.json / index.html      # Cloud deployment configurations
└── README.md                     # Project overview and documentation
```

---

## 🚀 Running Locally

### 1. Run the Web Application
```bash
cd website
python -m http.server 8080
```
Open `http://localhost:8080/index.html` in any modern web browser.

### 2. Train and Validate the Neural Network
```bash
python train_and_evaluate_all.py
```

### 3. Run the Verification & Parity Test Suites
```bash
python tests/test_model_verification.py
python tests/check_js_parity.py
```
*(All 5 verification tests pass, and Python-to-JavaScript parity is confirmed with 0.00000000 maximum discrepancy).*
