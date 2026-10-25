"""
Final Inference and Explainability Demo
=======================================
Runs the Experiment 3 end-to-end inference pipeline and prints human-readable
explanations for the top suspicious entities.
"""

import sys
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.detection.inference import run_inference
from src.detection.explain import explain_suspicious_entities

def main():
    print("Starting Inference Pipeline...")
    try:
        df = run_inference("../dkdata.json", checkpoint_path="artifacts/models/experiment3_graphsage.pt")
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    print("\nGenerating Explanations for Top Entities...\n")
    report = explain_suspicious_entities(df, top_k=5)
    print(report)

if __name__ == "__main__":
    main()
