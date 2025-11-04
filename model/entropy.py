import numpy as np
from scipy.stats import entropy

ENTROPY_BASE = 2

def compute_entropy(probability_distribution):
    return entropy(probability_distribution, base=ENTROPY_BASE)

def compute_maximum_entropy(n):
    return entropy(np.ones(n) / n, base=ENTROPY_BASE)