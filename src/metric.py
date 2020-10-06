"""
    This script was made by soeque1 at 24/07/20.
    To implement code for metric (e.g. NLL loss).
"""

from pytorch_lightning.metrics.nlp import BLEUScore

bleuS = BLEUScore(n_gram=4, smooth=True)
