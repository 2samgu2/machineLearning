import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from JSAnimation import IPython_display


np.random.seed(13)
data = np.random.random(100)

plt.hist(data, bins=15, normed=True, color='black', alpha=0.5)
plt.title('Histogram of $U(0,1)$ samples')
plt.show()

def normal(x, mu, sigma):
    """Normal distribution PDF."""
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
    
    
    
def _fit_gmm(data, num_components, num_iters=100):
    """Fit a single GMM with EM with one random initialization."""

    # Random initialization.
    mu = np.random.choice(data, num_components, replace=False)
    sigma = (np.random.random(num_components) * 0.15) + 0.1
    prob_population = np.ones(num_components) / num_components

    # Keep track of results after each iteration.
    results = []
    results.append((mu.copy(), sigma.copy(), prob_population.copy(), float('-inf')))
    
    last_log_likelihood = None
    for i in range(num_iters):
        # E-step.
        probs = [normal(x, mu, sigma) for x in data]
        probs = np.array([p / p.sum() for p in probs])

        # M-step.
        for k in range(num_components):
            k_probs = probs[:, k]
            mu[k] = (k_probs * data).sum() / k_probs.sum()
            sigma[k] = np.sqrt((k_probs * (data - mu[k])**2).sum() / k_probs.sum())
            prob_population[k] = k_probs.sum() / len(data)

        # Bookkeeping.
        log_likelihood = np.log(np.product([(normal(data[n], mu, sigma) * probs[n, :]).sum()
                                            for n in range(len(data))]))
        results.append((mu.copy(), sigma.copy(), prob_population.copy(), log_likelihood))
        if last_log_likelihood is not None and log_likelihood <= (last_log_likelihood + 0.01):
            break
        last_log_likelihood = log_likelihood

    return results
    
    
def fit_gmm(data, num_components, num_iters=10, num_random_inits=10):
    """Find the maximum likelihood GMM over several random initializations."""
    best_results = None
    best_results_iters = None
    best_ll = float('-inf')

    # Try several random initializations and keep the best.
    for attempt in range(num_random_inits):
        results_iters = _fit_gmm(data, num_components, num_iters=num_iters)
        final_log_likelihood = results_iters[-1][3]
        if final_log_likelihood > best_ll:
            best_results = results_iters[-1]
            best_results_iters = results_iters
            best_ll = final_log_likelihood

    return best_results, best_results_iters
    
    
    
colors = 'bgrcmy'

def gmm_fit_and_animate(data, num_components, interval=200):
    _, best_results_iters = fit_gmm(data, num_components, num_iters=200, num_random_inits=10)

    # Remove initial random guess (before doing a single iteration).
    best_results_iters = best_results_iters[1:]
    
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(xlim=(0, 1), ylim=(0, 2))
    
    line, = ax.plot([], [], label='GMM Fit', color='black', alpha=0.7, linewidth=3)
    
    ax.hist(data, normed=True, bins=15, color='lightgray', alpha=0.2, label='Real Data')
    ax.legend()
    ax.set_title('{0} Components'.format(num_components))

    def animate(i):
        mu, sigma, prob_population, _ = best_results_iters[i]
        xs = np.linspace(0, 1, 1000)
        ys = [(normal(x, mu, sigma) * prob_population).sum() for x in xs]
        line.set_data(xs, ys)
        
        for k in range(num_components):
            ys = [normal(x, mu[k], sigma[k]) * prob_population[k] for x in xs]
            ax.plot(xs, ys, alpha=0.2, color=colors[k % len(colors)])

    # Things like to crash if I try to do too many frames, I guess, so limit
    # the number of frames.
    num_iters = len(best_results_iters)
    frames = np.arange(0, num_iters, max(1, num_iters // 20), dtype=int)

    return animation.FuncAnimation(fig, animate, frames=frames, interval=interval)
    
gmm_fit_and_animate(data, 3)
