# Inside Diabetes — Cinematic Medical Documentary & Explainable AI Prediction

An immersive, scroll-driven 3D biological exploration and explainable artificial intelligence (XAI) risk prediction platform for diabetes.

## Overview

- **3D Cinematic Biology:** Built with Three.js featuring continuous Steadicam camera physics, realistic anatomical models (Full Human Anatomy, Vascular Bloodstream with Erythrocytes, Leukocytes & Glucose, Pancreatic Islets, and Cellular Insulin Receptor Signaling).
- **Explainable Neural Network:** Trained on the authentic NIDDK Pima Indians Diabetes Database using PyTorch/Scikit-Learn, with real-time SHAP feature attribution.
- **Minimal Tooltips & Zero Text Overlap:** Concise 1-2 sentence medical explanations on object hover/tap.
- **Interactive Risk Portal:** Real-time patient risk calculator with clinical threshold telemetry.

## Project Structure

```
Diabetes_AI_Project/
├── data/
│   └── diabetes.csv          # Pima Indians Diabetes Database (NIDDK)
├── model/
│   ├── diabetes_ann.pkl      # Trained PyTorch/Scikit-Learn model
│   ├── model_weights.json    # Exported weights & biases for in-browser inference
│   ├── scaler.pkl            # Feature standardizer
│   └── train_model.py        # ANN training and SHAP analysis script
└── website/
    ├── index.html            # Main documentary website & prediction portal
    ├── style.css             # Cinematic CSS design system
    ├── script.js             # Three.js 3D biology & ANN inference engine
    └── lib/                  # Local Three.js & GSAP libraries
```

## Running Locally

To run the platform locally:

```bash
cd website
python -m http.server 8080
```

Open `http://localhost:8080/index.html` in your browser.
