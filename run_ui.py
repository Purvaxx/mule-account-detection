"""
UI Presentation Layer Entry Point
=================================

Starts the interactive hackathon AML dashboard.
This purely reads from the frozen ML detection pipeline and the persisted
Experiment 3 checkpoint.
"""

import sys
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.ui.app import run_server

if __name__ == "__main__":
    run_server()
