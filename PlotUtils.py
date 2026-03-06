import matplotlib.pyplot as plt

#Visualizes how opinions change over time
#TO-DO Delete if it is not needed at end. plot_evolution_comparison better
def plot_evolution(history, title="Opinion Evolution", sample_size=None):
    """    
    Parameters:
    history        — np.array that contains the history
    title          — plot title
    sample         — Max number of agents history to plot
    """
    alpha=0.4
    line_width=0.8
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    n_agents = history.shape[1]
    indices = range(n_agents)
    
    # Select random set of agents 
    if sample and sample < n_agents:
        indices = np.random.choice(n_agents, sample_size, replace=False)
    
    for i in indices:
        ax.plot(history[:, i], alpha=alpha, linewidth=line_width)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Opinion')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

#Visualizes how different evolutions compare. Useful for comparing the results of a given experiment
def plot_evolution_comparison(histories, titles, suptitle=""):
    """
    Parameters:
    -----------
    histories        — list of hitories of the simulations
    titles           — list of titles for each plot
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

