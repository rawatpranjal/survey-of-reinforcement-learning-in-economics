import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS_DIR = os.path.dirname(HERE)
if SIMS_DIR not in sys.path:
    sys.path.insert(0, SIMS_DIR)
