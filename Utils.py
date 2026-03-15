import numpy as np
import matplotlib.pyplot as plt

"""
This module contains utility functions that does not necesseruly has much in common, but would be overcomplication to separate them into files.
"""

def get_clusters(opinions, epsilon=0.5):
    """
    Identify opinion clusters based on a confidence bound.
    First we sort the values. Then we count separate cluster if the difference between the max value of the current and the next min value is > than epsilon. 
    """
    sorted_opinions = sorted(list(opinions)) 
    clusters = []
    current_cluster = [sorted_opinions[0]]
    
    for i in range(1, len(sorted_opinions)):
        if sorted_opinions[i] - sorted_opinions[i-1] > epsilon:
            #The difference is bigger than epsilon so we start counting another cluster
            clusters.append(current_cluster)
            current_cluster = [sorted_opinions[i]]
        else:
            # Difference is less tan epsilon so we are still in the same cluster
            current_cluster.append(sorted_opinions[i])
    
    clusters.append(current_cluster)
    
    return clusters


def get_polarization_index(opinions):
    """
    Returns polarization index based on the variance
    """
    variance = np.var(opinions)

    # We divide by 0.25 because it is the maximum variance any distribution on real numbers from 0 to 1
    # By dividing we ensure that our index is a number between 0 and 1
    normalized_variance = variance/0.25 
    
    return normalized_variance



def get_theoretical_number_of_clusters(epsilon):
    """
    Returns the theoretical number of cluster, based on the formula:
    clusters = (approx.) 1 / 2*epsilon
    """
    return int(1 / (2 * epsilon))
               

def epsilon_sweep(opinions_list, epsilon_values):
    """
    Compute simulation metrics for multiple epsilon values
    """
    results = []

    for opinions, eps in zip(opinions_list, epsilon_values):
        results.append(
            [eps,
             len(get_clusters(opinions, eps)),
             get_polarization_index(opinions),
            get_theoretical_number_of_clusters(eps)
            ])

    return results

def average_epsilon_sweep(epsilon_values, opinions_dict):
    """
    Compute simulation metrics for multiple epsilon values over multiple runs
    """
    result = []

    for eps in epsilon_values:
        opinions        = opinions_dict[eps]
        clusters        = np.mean([len(get_clusters(op, eps)) for op in opinions])
        polarization    = np.mean([get_polarization_index(op) for op in opinions])
        theoretical     = get_theoretical_number_of_clusters(eps)

        result.append([eps, clusters, polarization, theoretical])

    return result

def plot_epsilon_sweep(results, title=""):
    """
    Plot simulation results from an epsilon sweep.
    Two plots are generated:
    - Number of clusters vs epsilon (simulated vs theoretical)
    - Polarization index vs epsilon

    It works specifically with the format data epsilon_sweep and average_epsilon_sweep return.
    """
    epsilon_list        = [r[0] for r in results]
    simulated_list      = [r[1] for r in results]
    polarization_list   = [r[2] for r in results]
    theoretical_list    = [r[3] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), clear=True)

    axes[0].plot(epsilon_list, simulated_list,   'o-',  label='Simulated')
    axes[0].plot(epsilon_list, theoretical_list, 's--', label='Theory')
    axes[0].set_xlabel('Epsilon')
    axes[0].set_ylabel('Clusters')
    axes[0].legend()

    axes[1].plot(epsilon_list, polarization_list, 'o-', color='red')
    axes[1].set_xlabel('Epsilon')
    axes[1].set_ylabel('Polarization Index')

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


#Visualizes how different evolutions compare. Useful for comparing the results of a given experiment
def plot_evolution_comparison(histories, titles, suptitle=""):
    """
    Parameters:
    -----------
    histories        - list of hitories of the simulations
    titles           - list of titles for each plot
    """
    alpha=0.4
    line_width=0.8
    font_size=13
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)
    
    if n == 1:
        axes = [axes]
    
    for ax, history, title in zip(axes, histories, titles):
        n_agents = history.shape[1]
        for i in range(n_agents):
            ax.plot(history[:, i], alpha=alpha, linewidth=line_width)
        ax.set_title(title)
        ax.set_xlabel('Time Step')
        ax.set_ylim(-0.05, 1.05)
    
    axes[0].set_ylabel('Opinion')
    
    if suptitle:
        plt.suptitle(suptitle, fontsize=font_size)
    
    plt.tight_layout()
    plt.show()