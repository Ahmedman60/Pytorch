import random
import copy

# Genetic Programming Example: Evolving a Function to Approximate f(x) = x + 3

# Define a simple tree structure for Genetic Programming


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        """Evaluate the tree using input x."""
        if self.value == '+':
            return self.left.evaluate(x) + self.right.evaluate(x)
        elif self.value == '-':
            return self.left.evaluate(x) - self.right.evaluate(x)
        elif self.value == '*':
            return self.left.evaluate(x) * self.right.evaluate(x)
        elif self.value == '/':
            return self.left.evaluate(x) / self.right.evaluate(x) if self.right.evaluate(x) != 0 else 1
        else:
            return x if self.value == 'x' else float(self.value)

    def copy(self):
        """Return a deep copy of the tree."""
        return copy.deepcopy(self)

    def print(self, level=0):
        """Recursively print the tree in a human-readable format."""
        indent = "    " * level
        if self.left and self.right:
            print(f"{indent}{self.value}")
            self.left.print(level + 1)
            self.right.print(level + 1)
        else:
            print(f"{indent}{self.value}")

# Generate a random tree representing a mathematical function


def generate_random_tree(depth=3):
    if depth == 0:
        return Node(random.choice(['x', str(random.randint(1, 10))]))
    op = random.choice(['+', '-', '*', '/'])
    return Node(op, generate_random_tree(depth - 1), generate_random_tree(depth - 1))

# Perform subtree crossover to create new functions


def crossover(parent1, parent2):
    """Swap random subtrees between two parents."""
    child1, child2 = parent1.copy(), parent2.copy()

    def get_random_subtree(node):
        if random.random() < 0.3 or (node.left is None and node.right is None):
            return node
        return get_random_subtree(node.left if random.random() < 0.5 else node.right)

    subtree1 = get_random_subtree(child1)
    subtree2 = get_random_subtree(child2)
    subtree1.value, subtree1.left, subtree1.right = subtree2.value, subtree2.left, subtree2.right

    return child1, child2

# Tournament selection to pick the best individuals


def tournament_selection(population, fitnesses, t=7):
    """Select the best individual from a random subset of the population."""
    tournament = random.sample(list(zip(population, fitnesses)), t)
    return max(tournament, key=lambda x: x[1])[0]

# Fitness function (Measures how close the tree approximates f(x) = x + 3)


def fitness(individual):
    # Trying to approximate y = x + 3
    test_cases = [(x, x + 3) for x in range(-10, 10)]
    error = sum(abs(individual.evaluate(x) - y) for x, y in test_cases)
    return -error  # Lower error = higher fitness

# Genetic Programming Algorithm to evolve a function


def genetic_programming(popsize=10, max_generations=50):
    """Main function that runs the Genetic Programming algorithm."""
    print("Starting Genetic Programming to evolve a function approximating f(x) = x + 3...")
    population = [generate_random_tree() for _ in range(popsize)]
    best_individual = None

    for generation in range(max_generations):
        fitnesses = [fitness(ind) for ind in population]
        best_individual = max(population, key=fitness)
        print(
            f"Generation {generation + 1}: Best Fitness = {-fitness(best_individual)}")

        new_population = []
        while len(new_population) < popsize:
            if random.random() < 0.9:  # 90% of the time perform crossover
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
                child1, child2 = crossover(parent1, parent2)
                new_population.append(child1)
                if len(new_population) < popsize:
                    new_population.append(child2)
            else:  # 10% of the time perform direct copy
                parent = tournament_selection(population, fitnesses)
                new_population.append(parent.copy())

        population = new_population  # Update the population

    print("Evolution completed. Best function found:")
    return best_individual


# Run the Genetic Programming Algorithm
best_tree = genetic_programming()
print("\nBest Tree Structure:")
best_tree.print()
print("\nBest Tree Function Evaluation at x=5:", best_tree.evaluate(5))
