"""Optional paired statistical comparisons on independent experimental units."""
import itertools
import numpy as np

def paired_sign_flip(differences, seed=42, repetitions=10000):
    """Two-sided test of a mean paired difference under sign exchangeability."""
    d = np.asarray(differences, dtype=float)
    if len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Comparisons require at least two independent, finite unit differences")
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    if len(d) <= 16:
        statistics = (abs(np.dot(signs, d) / len(d))
                      for signs in itertools.product([-1, 1], repeat=len(d)))
        extreme = sum(s >= observed - 1e-15 for s in statistics)
        p = extreme / 2**len(d)
        method = "exact paired sign-flip"
    else:
        extreme = 0
        for _ in range(repetitions):
            extreme += abs(np.mean(d * rng.choice([-1, 1], len(d)))) >= observed - 1e-15
        p = (extreme + 1) / (repetitions + 1)
        method = "Monte Carlo paired sign-flip"
    means = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(repetitions)]
    return {"units": len(d), "mean_difference_a_minus_b": float(d.mean()),
            "bootstrap_95_percent_interval": np.quantile(means, [.025, .975]).tolist(),
            "p_value": p, "method": method, "seed": seed, "repetitions": repetitions}

def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(order))
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(order) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted.tolist()
