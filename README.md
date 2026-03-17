# Opinion Dynamics

A mathematical exploration of how opinions form and polarize in human populations. Exploring how the level of tolerance, the topology and biases influence these processes.

## Description

This project implements and analyses two classical opinion dynamics models -
Deffuant and Hegselmann-Krause. Examines  the level of
tolerance, network topology and hub influence shape the final opinion
distribution of a population.

## Key Questions
- Under what conditions does a population reach consensus vs. fragment into clusters?
- How does network topology (who can talk to whom) affect the outcome?
- Do highly connected agents disproportionately shape collective opinion?

## Structure

- `MainProjectNotebook.ipynb` - main notebook with all experiments and analysis
- `ModelsImplementations.py` - Deffuant and HK model implementations
- `Graph.py` - graph class and network generators (fully connected, small-world, scale-free)
- `Utils.py` - helper functions for metrics and plotting
