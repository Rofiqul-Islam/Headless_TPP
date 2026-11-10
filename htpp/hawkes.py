"""Multivariate Hawkes processes with exponential kernels (synthetic data with a
known likelihood):  lambda_k(t) = mu_k + sum_{t_j < t} alpha[k, k_j] beta exp(-beta (t - t_j))."""
from __future__ import annotations

import numpy as np


def simulate(mu, alpha, beta, T, rng):
    """Ogata thinning on [0, T]. Returns (times, marks)."""
    mu = np.asarray(mu, float)
    alpha = np.asarray(alpha, float)
    K = len(mu)
    a = np.zeros(K)
    t = 0.0
    times, marks = [], []
    while True:
        lam_bar = mu.sum() + a.sum()       # intensity only decays until the next event
        dt = rng.exponential(1.0 / lam_bar)
        if t + dt > T:
            break
        t += dt
        a = a * np.exp(-beta * dt)
        lam = mu + a
        if rng.random() * lam_bar <= lam.sum():
            k = rng.choice(K, p=lam / lam.sum())
            times.append(t)
            marks.append(k)
            a = a + alpha[:, k] * beta
    return np.array(times), np.array(marks, dtype=np.int64)


def loglik(mu, alpha, beta, times, marks, T, t0=0.0):
    """Exact log-likelihood of the events in (t0, T] given all earlier events, where
    the events at times <= t0 are part of the history (not scored)."""
    mu = np.asarray(mu, float)
    alpha = np.asarray(alpha, float)
    K = len(mu)
    a = np.zeros(K)
    last = 0.0
    ll = 0.0
    for t, k in zip(times, marks):
        a = a * np.exp(-beta * (t - last))
        if t > t0:
            ll += np.log(mu[k] + a[k])
        a = a + alpha[:, k] * beta
        last = t
    # compensator over (t0, T]
    comp = mu.sum() * (T - t0)
    if len(times):
        times = np.asarray(times)
        marks = np.asarray(marks)
        start = np.maximum(times, t0)
        decay = np.exp(-beta * (start - times)) - np.exp(-beta * (T - times))
        comp += (alpha[:, marks].sum(axis=0) * decay).sum()
    return ll - comp


def default_params(K: int, rng: np.random.Generator, branching_ratio: float = 0.6, mean_rate: float = 1.0):
    """Random stable parameters: baseline rates mu, excitation matrix alpha scaled to the
    given spectral radius (branching ratio), decay beta."""
    mu = rng.uniform(0.1, 0.5, K)
    mu = mu / mu.sum() * mean_rate * (1 - branching_ratio)
    alpha = rng.uniform(0.0, 1.0, (K, K)) * (rng.random((K, K)) < 0.7)
    alpha += np.eye(K) * 0.5
    rho = max(abs(np.linalg.eigvals(alpha)))
    alpha = alpha / rho * branching_ratio
    beta = 1.0
    return mu, alpha, beta
