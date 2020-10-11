"""
    This script was made by soeque1 at 24/07/20.
    To implement code for metric (e.g. NLL loss).
"""

import torch
from pytorch_lightning.metrics.functional.nlp import bleu_score


class BLEUScore:
    """
    Calculate BLEU score of machine translated text with one or more references.

    Example:

        >>> translate_corpus = ['the cat is on the mat'.split()]
        >>> reference_corpus = [['there is a cat on the mat'.split(), 'a cat is on the mat'.split()]]
        >>> metric = BLEUScore()
        >>> metric(translate_corpus, reference_corpus)
        tensor(0.7598)
    """

    def __init__(self, n_gram: int = 4, smooth: bool = False):
        """
        Args:
            n_gram: Gram value ranged from 1 to 4 (Default 4)
            smooth: Whether or not to apply smoothing – Lin et al. 2004
        """
        self.n_gram = n_gram
        self.smooth = smooth

    def forward(self, translate_corpus: list, reference_corpus: list) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            translate_corpus: An iterable of machine translated corpus
            reference_corpus: An iterable of iterables of reference corpus

        Return:
            torch.Tensor: BLEU Score
        """
        return bleu_score(
            translate_corpus=translate_corpus,
            reference_corpus=reference_corpus,
            n_gram=self.n_gram,
            smooth=self.smooth,
        ).to(self.device, self.dtype)


bleuS = BLEUScore(n_gram=4, smooth=True)
