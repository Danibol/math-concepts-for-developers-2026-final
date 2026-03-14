import numpy as np
import matplotlib.pyplot as plt
from Graph import Graph

CONVERGENCE_TOL = 1e-2
CONVERGENCE_CHECK_EVERY = 1000
#TO-DO THink if convergence check makese sense from efficiency pov
#TO-DO Document code...
def generate_opinions(agents_number, opinion_seed):
    return np.random.default_rng(opinion_seed).uniform(0, 1, agents_number)

def _allocate_history(steps, agents_number, initial_opinions):
    history    = np.empty((steps + 1, agents_number))
    history[0] = initial_opinions
    return history
    

def _deffuant_step(opinions, rng, epsilon, mu, graph=None):
    if graph is not None:
        i, j = graph.get_random_edge(rng)
    else:
        i, j = rng.choice(len(opinions), 2, replace=False)
    diff = opinions[j] - opinions[i]
    if abs(diff) < epsilon:
        delta = mu * diff
        opinions[i] += delta
        opinions[j] -= delta


def _hk_step(opinions, epsilon, graph=None):
    
    if graph is not None:
        new_opinions = opinions.copy()
        for i in range(len(opinions)):
            # get all the candidates
            #include i because we will have to take its own opinion when averaging
            candidates = np.append(graph.neighbors(i), i)

            # get the actual opinions of the candidates
            local = opinions[candidates]

            # Determine which of the candidates have opinion close enough to i
            mask = np.abs(local - opinions[i]) <= epsilon

            # Update the opinion of i to be equal to the mean of all candidates that has opinion close enough to its own
            new_opinions[i] = local[mask].mean()

        return new_opinions

    # Compute pairwise distance matrix - shape (n, n)
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
    interaction_seed=None,
    graph=None
):
    opinions = generate_opinions(agents_number, opinion_seed)
    rng = np.random.default_rng(interaction_seed)
    prev = opinions.copy()

    for step in range(steps):   
        _deffuant_step(opinions, rng, epsilon, mu, graph)
        if step % CONVERGENCE_CHECK_EVERY == 0 and step > 0:
            diff = np.max(np.abs(opinions - prev))

            #print(f'step={step}, max_diff={diff:.2e}')
            if diff < CONVERGENCE_TOL:
                #print(f'converged at step {step}')
                break
            prev = opinions.copy()
    return opinions

def deffuant_history(agents_number=100, epsilon=0.3, mu=0.5, steps=50000,
                     opinion_seed=None, interaction_seed=None, graph = None):
    opinions = generate_opinions(agents_number, opinion_seed)
    rng      = np.random.default_rng(interaction_seed)
    history  = _allocate_history(steps, agents_number, opinions)
    for t in range(steps):
        _deffuant_step(opinions, rng, epsilon, mu, graph)
        history[t+1] = opinions
    return history


# HK
def hk(
    agents_number=100,
    epsilon=0.3,
    steps=100,
    opinion_seed=None,
    graph = None
):
    opinions = generate_opinions(agents_number, opinion_seed)
    for step in range(steps):
        opinions = _hk_step(opinions, epsilon, graph)
        
    return opinions

def hk_history(agents_number=100, epsilon=0.3, steps=100, opinion_seed=None, graph = None):
    opinions = generate_opinions(agents_number, opinion_seed)
    history  = allocate_history(steps, agents_number, opinions)
    for t in range(steps):
        opinions     = _hk_step(opinions, epsilon, graph)
        history[t+1] = opinions
    return history

def hk_history(agents_number=100, epsilon=0.3, steps=100,
               opinion_seed=None, graph=None, initial_opinions=None):
    if initial_opinions is not None:
        opinions = initial_opinions.copy()
    else:
        generate_opinions(agents_number, opinion_seed)

    history  = _allocate_history(steps, agents_number, opinions)
    for t in range(steps):
        opinions     = _hk_step(opinions, epsilon, graph)
        history[t+1] = opinions
    return history