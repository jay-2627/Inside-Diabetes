# RESEARCH_PROBLEM.md

## Project Context
The **Inside Diabetes** platform provides an immersive 3D interactive biological exploration of human anatomy alongside an in-browser Explainable Artificial Neural Network (ANN) that predicts diabetes risk using the NIDDK Pima Indians dataset.

## Research Goal
Deliver a **clinically reliable, well-calibrated, explainable neural network platform** that:
1. Deploys an optimized Deep Multilayer Perceptron (MLP: $8 \to 32 \to 16 \to 1$, Sigmoid).
2. Minimizes clinical False Negatives through validation-derived screening threshold optimization ($\theta^* = 0.25$, Sensitivity $85.2\%$).
3. Supplies authentic **Integrated Gradients** feature attributions for individual patient predictions.
4. Enables **interactive counterfactual exploration** with immutable feature locking.
5. Preserves a zero-latency, server-independent client-side inference engine paired with cinematic 3D biological visuals.
