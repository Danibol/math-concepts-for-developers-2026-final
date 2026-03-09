import numpy as np
import matplotlib.pyplot as plt

CONVERGENCE_TOL = 1e-6
CONVERGENCE_CHECK_EVERY = 100
#TO-DO THink if convergence check makese sense from efficiency pov
#TO-DO Document code...
def _generate_opinions(agents_number, opinion_seed):
    return np.random.default_rng(opinion_seed).uniform(0, 1, agents_number)

def _allocate_history(steps, agents_number, initial_opinions):
    history    = np.empty((steps + 1, agents_number))
    history[0] = initial_opinions
    return history

def _deffuant_step(opinions, rng, epsilon, mu):
    i, j = rng.choice(len(opinions), 2, replace=False)
    diff = opinions[j] - opinions[i]
    if abs(diff) < epsilon:
        delta = mu * diff
        opinions[i] += delta
        opinions[j] -= delta


def _hk_step(opinions, epsilon):
    # Compute pairwise distance matrix — shape (n, n)
    distances = np.abs(opinions[:, None] - opinions[None, :])
    
    # Boolean mask - True where two agents are within confidence bound
    neighbors = distances <= epsilon
    
    # Each agent moves to the average opinion of its neighbors
    return (neighbors * opinions).sum(axis=1) / neighbors.sum(axis=1)


# Deffuant
def deffuant(
    agents_number=100,
    epsilon=0.3,
    mu=0.5,
    steps=50000,
    opinion_seed=None,
    interaction_seed=None
):
    opinions = _generate_opinions(agents_number, opinion_seed)
    rng = np.random.default_rng(interaction_seed)

    for step in range(steps):
        _deffuant_step(opinions, rng, epsilon, mu)

    return opinions

def deffuant_history(agents_number=100, epsilon=0.3, mu=0.5, steps=50000,
                     opinion_seed=None, interaction_seed=None):
    opinions = _generate_opinions(agents_number, opinion_seed)
    rng      = np.random.default_rng(interaction_seed)
    history  = _allocate_history(steps, agents_number, opinions)
    for t in range(steps):
        _deffuant_step(opinions, rng, epsilon, mu)
        history[t+1] = opinions
    return history


# HK
def hk(
    agents_number=100,
    epsilon=0.3,
    steps=100,
    opinion_seed=None
):
    opinions = _generate_opinions(agents_number, opinion_seed)
    for step in range(steps):
        opinions = _hk_step(opinions, epsilon)
        
    return opinions

def hk_history(agents_number=100, epsilon=0.3, steps=100, opinion_seed=None):
    opinions = _generate_opinions(agents_number, opinion_seed)
    history  = _allocate_history(steps, agents_number, opinions)
    for t in range(steps):
        opinions     = _hk_step(opinions, epsilon)
        history[t+1] = opinions
    return history