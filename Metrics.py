import numpy as np
import matplotlib.pyplot as plt

# TO-DO check all function for improvements, check for better plotting ideas
def get_clusters(opinions, epsilon=0.5):
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
    variance = np.var(opinions)

    # We divide by 0.25 because it is the maximum variance any distribution on real numbers from 0 to 1
    # By dividing we ensure that our index is a number between 0 and 1
    normalized_variance = variance/0.25 
    
    return normalized_variance


def get_theoretical_number_of_clusters(epsilon):
    return int(1 / (2 * epsilon))
               
# Prints summarized data about a set of opinions
def print_summary(opinions, epsilon, title=""):
    clusters = get_clusters(opinions, epsilon)
    polarization = get_polarization_index(opinions)
    cluster_means = [np.mean(c) for c in clusters]
    cluster_sizes = [len(c) for c in clusters]
    
    print(f"--- {title} ---")
    print(f"  Agents:          {len(opinions)}")
    print(f"  Epsilon:         {epsilon}")
    print(f"  Clusters:        {len(clusters)}")
    print(f"  Centers:         {cluster_means}")
    print(f"  Sizes:           {cluster_sizes}")
    print(f"  Polarization:    {polarization}")
    print()

def epsilon_sweep(opinions_list, epsilon_values):
    results = []

    for opinions, eps in zip(opinions_list, epsilon_values):
        results.append(
            [eps,
             len(get_clusters(opinions, eps)),
             get_polarization_index(opinions),
            get_theoretical_number_of_clusters(eps)
            ])

    return results

def plot_epsilon_sweep(results, title=""):
    epsilon_list        = [r[0] for r in results]
    simulated_list      = [r[1] for r in results]
    polarization_list   = [r[2] for r in results]
    theoretical_list    = [r[3] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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