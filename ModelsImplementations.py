import numpy as np
import matplotlib.pyplot as plt
from Graph import Graph


# Constants used when determining if Deffuant/HK has converged
CONVERGENCE_TOL = 1e-2
CONVERGENCE_CHECK_EVERY = 1000


def generate_opinions(agents_number, opinion_seed):
    """Generates uniformly distributed numbers in the interval [0,1]"""
    return np.random.default_rng(opinion_seed).uniform(0, 1, agents_number)

def _allocate_history(steps, agents_number, initial_opinions):
    """
    Returns array for storing how opinions change over time.
    It is used to allocate memory beforehand in order to save some time from allocating at each step.
    """
    history    = np.empty((steps + 1, agents_number))
    history[0] = initial_opinions
    return history
    

def _deffuant_step(opinions, rng, epsilon, mu, graph=None):
    """
    This is the step for Deffuant algorithm implementation.
    Selects a random pair of agents and updates their opinions if they are
    within the confidence bound epsilon.
    If graph is specified, then the pair can only consist of neighbours. 
    """
    if graph is not None:
        i, j = graph.get_random_edge(rng)
    else:
        i, j = rng.choice(len(opinions), 2, replace=False)

    
    diff = opinions[j] - opinions[i]
    if abs(diff) < epsilon:
        # Update opinions in the way the Deffuant model requires. DOne only if the difference is withing the confidence bound Epsilon
        delta = mu * diff
        opinions[i] += delta
        opinions[j] -= delta


def _hk_step(opinions, epsilon, graph=None):
     """
    Each agent is moved to the mean of the opinions of all agents withing the confidence bound.
    Optionally Graph can be specified so an agent can be directly influence only by its neighbours.
    """
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
    """
    Runs the Deffuant model and returns only the final opinions. We dont keep full history here, so it is a bit faster.
    Used in the experiments when no opinion history is needed.
    """
    opinions = generate_opinions(agents_number, opinion_seed)
    rng = np.random.default_rng(interaction_seed)
    prev = opinions.copy()

    for step in range(steps):   
        _deffuant_step(opinions, rng, epsilon, mu, graph)

        # Check if the process has converged in order to save some time and not run 
        # more steps than needed. This function only needs the final opinions, so after convergence there is no need to continue.
        # The check is done every CONVERGENCE_CHECK_EVERY steps. Too often still makes it slow. SO current value is balanced
        if step % CONVERGENCE_CHECK_EVERY == 0 and step > 0:
            # Calculate the maximal change of opinions since the last check
            diff = np.max(np.abs(opinions - prev))

            # If there is no significat change, break
            if diff < CONVERGENCE_TOL:
                break

            # Save the current opinions as they are needed for the next check
            prev = opinions.copy()
    return opinions


def deffuant_history(agents_number=100, epsilon=0.3, mu=0.5, steps=50000,
                     opinion_seed=None, interaction_seed=None, graph = None):
    """
    Does the same as the deffuant function, but returns the whole history of opinions.
    Separated them in order to be easier to know what format of result to expect.
    """
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
    """
    Run the HK model. Returns only final opinions.
    No convergence check as it takes significantly less steps than Deffuant.
    It may make some sense, but not as much.
    """
    opinions = generate_opinions(agents_number, opinion_seed)
    for step in range(steps):
        opinions = _hk_step(opinions, epsilon, graph)
        
    return opinions
    

def hk_history(agents_number=100, epsilon=0.3, steps=100,
               opinion_seed=None, graph=None, initial_opinions=None):
    """
    Runs HK, but returns the full history.
    Initial opinions can be specified.
    """
    if initial_opinions is not None:
        opinions = initial_opinions.copy()
    else:
        opinions = generate_opinions(agents_number, opinion_seed)

    history  = _allocate_history(steps, agents_number, opinions)
    for t in range(steps):
        opinions     = _hk_step(opinions, epsilon, graph)
        history[t+1] = opinions
    return history