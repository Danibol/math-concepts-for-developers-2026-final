import numpy as np
import matplotlib.pyplot as plt


# Deffuant model
def deffuant(agents_number=100, epsilon=0.3, mu=0.5, steps=50000, seed=None):
    """
    Parameters:
    agents_number   — number of agents
    epsilon         — confidence threshold (0 to 1)
    mu              — convergence speed (0 to 0.5)
    steps           — number of steps
    seed            — random seed

    Returns:
    history         — opinion of each agent at each step
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random opinions
    opinions = np.random.uniform(0, 1, agents_number)
    history = [opinions.copy()]

    for _ in range(steps):
        # Pick two random agents
        i, j = np.random.choice(agents_number, 2, replace=False)

        # Check if opinions are close enough
        if abs(opinions[i] - opinions[j]) < epsilon:
            # They interact, so update values
            diff = opinions[j] - opinions[i]
            opinions[i] += mu * diff
            opinions[j] -= mu * diff

        history.append(opinions.copy())

    return np.array(history)


#Hegselmann-Krause (HK) Model
def hk(agents_number=100, epsilon=0.3, steps=100, seed=None):
    """
    Hegselmann-Krause Model.

    Parameters:
    agents_number      — number of agents
    epsilon            — confidence threshold (0 to 1)
    steps              — number of steps
    seed               — random seed 

    Returns:
    history            — opinion of each agent at each step
    """
    if seed is not None:
        np.random.seed(seed)

    #Generate random opinions
    opinions = np.random.uniform(0, 1, agents_number)
    history = [opinions.copy()]

    for _ in range(steps):
        new_opinions = np.zeros(agents_number)

        for i in range(agents_number):
            # Find all agents with opinion that is close enough
            neighbors = np.where(np.abs(opinions - opinions[i]) < epsilon)[0]
            # Update opinion to the mean
            new_opinions[i] = np.mean(opinions[neighbors])

        opinions = new_opinions
        history.append(opinions.copy())

    return np.array(history)