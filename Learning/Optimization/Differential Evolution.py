import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Import our differential evolution function
def differential_evolution(cost_func, bounds, population_size=20, F=0.8, CR=0.7, generations=100):
    dimensions = len(bounds)
    population = np.zeros((population_size, dimensions))
    for i in range(dimensions):
        population[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], size=population_size)
    
    fitness = np.zeros(population_size)
    for i in range(population_size):
        fitness[i] = cost_func(population[i])
    
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_score = fitness[best_idx]
    
    history = [best_score]
    
    for generation in range(generations):
        for i in range(population_size):
            candidates = list(range(population_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            mutant = population[a] + F * (population[b] - population[c])
            
            for j in range(dimensions):
                if mutant[j] < bounds[j][0]:
                    mutant[j] = bounds[j][0]
                if mutant[j] > bounds[j][1]:
                    mutant[j] = bounds[j][1]
            
            trial = population[i].copy()
            j_rand = np.random.randint(0, dimensions)
            for j in range(dimensions):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Normalize weights to sum to 1
            trial = trial / np.sum(trial)
            
            trial_fitness = cost_func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < best_score:
                    best_solution = trial.copy()
                    best_score = trial_fitness
        
        history.append(best_score)
        
    return best_solution, best_score, history

# Portfolio optimization example
# Let's define some assets with their expected returns and covariance matrix

# Number of assets
n_assets = 5

# Expected annual returns (in decimal form, e.g., 0.10 = 10%)
expected_returns = np.array([0.12, 0.10, 0.15, 0.08, 0.11])

# Covariance matrix (risk relationships between assets)
# Diagonal elements are variances, off-diagonal are covariances
covariance_matrix = np.array([
    [0.0400, 0.0050, 0.0060, 0.0030, 0.0020],
    [0.0050, 0.0250, 0.0040, 0.0025, 0.0015],
    [0.0060, 0.0040, 0.0500, 0.0045, 0.0035],
    [0.0030, 0.0025, 0.0045, 0.0160, 0.0010],
    [0.0020, 0.0015, 0.0035, 0.0010, 0.0290]
])

# Asset names for better display
asset_names = ["Tech Stocks", "Government Bonds", "Real Estate", "Corporate Bonds", "Commodities"]

# Define the objective function - maximize Sharpe ratio (return-to-risk ratio)
# We actually minimize the negative Sharpe ratio
def portfolio_objective(weights):
    # Normalize weights to ensure they sum to 1
    weights = weights / np.sum(weights)
    
    # Calculate portfolio return and risk
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    
    # Risk-free rate (e.g., Treasury bill)
    risk_free_rate = 0.02
    
    # Sharpe ratio (higher is better, so we negate for minimization)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    
    return -sharpe_ratio  # Negate because we want to maximize

# Define bounds (all weights between 0 and 1)
bounds = [(0, 1)] * n_assets

# Run differential evolution for portfolio optimization
best_weights, best_score, history = differential_evolution(
    portfolio_objective, 
    bounds, 
    population_size=50, 
    generations=100
)

# Normalize the weights
best_weights = best_weights / np.sum(best_weights)

# Calculate portfolio metrics
portfolio_return = np.sum(best_weights * expected_returns)
portfolio_risk = np.sqrt(np.dot(best_weights.T, np.dot(covariance_matrix, best_weights)))
sharpe_ratio = -best_score  # Convert back from minimization

print("\nPortfolio Optimization Results:")
print("------------------------------")
print(f"Expected Annual Return: {portfolio_return:.2%}")
print(f"Expected Annual Risk: {portfolio_risk:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print("\nOptimal Asset Allocation:")
for i in range(n_assets):
    print(f"{asset_names[i]}: {best_weights[i]:.2%}")

# Plot optimization progress
plt.figure(figsize=(10, 6))
plt.plot([-x for x in history])
plt.xlabel('Generation')
plt.ylabel('Sharpe Ratio')
plt.title('Portfolio Optimization Progress')
plt.grid(True)
plt.tight_layout()

# Plot asset allocation pie chart
plt.figure(figsize=(10, 7))
plt.pie(best_weights, labels=asset_names, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Optimal Portfolio Allocation')
plt.tight_layout()
plt.show()