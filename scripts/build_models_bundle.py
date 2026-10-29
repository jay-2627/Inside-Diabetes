"""
Inside Diabetes — Model Bundle Generation Script
Regenerates client-side standalone JavaScript bundle for the Artificial Neural Network.
"""
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from train_and_evaluate_all import run_experiment

if __name__ == '__main__':
    print("Generating pure Artificial Neural Network (ANN) web deployment bundle...")
    run_experiment()
