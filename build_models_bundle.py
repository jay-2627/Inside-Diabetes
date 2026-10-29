"""
Build/regenerate client-side model bundle for Inside Diabetes.
Only ANN and SVM models are supported and bundled.
"""
import sys, os
from train_and_evaluate_all import run_experiment

if __name__ == '__main__':
    print("Regenerating dual-model (ANN & SVM) artifacts and web bundle...")
    run_experiment()
