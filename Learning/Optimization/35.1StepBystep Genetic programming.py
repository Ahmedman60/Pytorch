import random
import copy


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        """Evaluate the tree using input x."""
        if self.value == '+':
            return 3 + 4
        elif self.value == '-':
            return self.left.evaluate(x) - self.right.evaluate(x)
        elif self.value == '*':
            return 2 * 2
        elif self.value == '/':
            return self.left.evaluate(x) / self.right.evaluate(x) if self.right.evaluate(x) != 0 else 1
        else:
            return x if self.value == 'x' else float(self.value)

    def copy(self):
        """Return a deep copy of the tree."""
        return copy.deepcopy(self)

    def print(self, level=0):
        """Recursively print the tree in a human-readable format."""
        indent = "   " * level
        if self.left and self.right:
            # start put the indentation in screen next to it the value.
            print(f"{indent}{self.value}")
            self.left.print(level + 1)
            self.right.print(level + 1)
        else:
            print(f"{indent}{self.value}")


def generate_random_tree(depth=2):
    if depth == 0:
        return Node(random.choice(['x', str(random.randint(1, 10))]))
    op = random.choice(['+', '-', '*', '/'])
    return Node(op, generate_random_tree(depth - 1), generate_random_tree(depth - 1))


def crossover(parent1, parent2):
    '''
With 30% probability, we select the current node.
Otherwise, we recursively move left or right (randomly) evenly to pick a deeper subtree.
If we reach a leaf node (like 'x' or a constant), we stop -- fix error.

    '''
    """Swap random subtrees between two parents."""
    child1, child2 = parent1.copy(), parent2.copy()

    def get_random_subtree(node):
        """Recursively select a random subtree."""
        if random.random() < 0.3 or (node.left is None and node.right is None):
            return node
        return get_random_subtree(node.left if random.random() < 0.5 else node.right)

    subtree1 = get_random_subtree(child1)
    subtree2 = get_random_subtree(child2)
    subtree1.value, subtree1.left, subtree1.right = subtree2.value, subtree2.left, subtree2.right

    return child1, child2


'''
The fitness function evaluates how good a given mathematical expression (tree) from what we get above --> is at solving a problem.
'''


def fitness(individual):
    '''
    Trying to approximate f(x) = x + 3
    i will build test-cases or datasets to train the model
    dataset will be in shape of (x,f(x))    then pass those numbers to the individual (tree you testing).
    calculate absolute error or MSE between (i prefare MSE or RMSE-but for simplesty i use AE) the f(x) and the output from tree evaluate.
    of course you can change the function  x+3 to anything you want.
    '''
    test_cases = [(x, x + 3) for x in range(-10, 10)]
    error = sum(abs(individual.evaluate(x) - y) for x, y in test_cases)
    '''
    why i use - here because the problem is maxmaization problem.  
    i will order the individual based on error i want the one with less error
    if errors were like [50,20,98,5]   if i ordered based on max  i get 98 which is the worest  but if i add negative  i will get -5 which is the largets.
    '''
    return -error


def tournament_selection(population, fitnesses, t=7):
    """Select the best individual from a random subset of the population."""
    tournament = random.sample(list(zip(population, fitnesses)), t)
    return max(tournament, key=lambda x: x[1])[0]  # Return best individual


# Testing out the Tournament selection.
'''
t=7 means each tournament selects 7 random individuals.
The higher t is, the stronger the selection pressure:
Small t (e.g., 2-3) → More exploration (diversity, weaker selection).
Large t (e.g., 7-10) → More exploitation (selects strong individuals more often).
Genetic Programming typically uses t=7 because it strongly favors better individuals while keeping some diversity.
(From the book..)
'''

# test_population = ["Tree1", "Tree2", "Tree3", "Tree4",
#                    "Tree5", "Tree6", "Tree7", "Tree8", "Tree9", "Tree10"]
# # tree6  is more likely to be selected because it have highest fitness (lowest error)
# test_fitnesses = [-10, -20, -15, -30, -25, -5, -40, -35, -45, -50]


# selected = [tournament_selection(
#     test_population, test_fitnesses, t=7) for _ in range(5)]

# # Print results
# print("Selected Individuals:", selected)


def genetic_programming(popsize=10, max_generations=50):
    population = [generate_random_tree() for _ in range(popsize)]
    best_individual = None

    for generation in range(max_generations):
        fitnesses = [fitness(ind) for ind in population]
        # if you don't get the below you can zip the population and fintess and sort by fitness then take the first individual like we did before.
        best_individual = max(population, key=fitness)

        new_population = []
        while len(new_population) < popsize:
            if random.random() < 0.9:  # 90% Crossover
                # same parent can be selected .
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
                child1, child2 = crossover(parent1, parent2)
                new_population.append(child1)
                if len(new_population) < popsize:
                    new_population.append(child2)
            else:  # 10% Direct Copy
                parent = tournament_selection(population, fitnesses)
                new_population.append(parent.copy())

        population = new_population
        print(
            f"Generation {generation + 1}: Best Fitness = {-fitness(best_individual)}")

    return best_individual


# The test of the end
best_tree = genetic_programming()
print("\nBest Tree Structure:")
best_tree.print()
# x+5=8  ^^ finally.
print("\nBest Tree Function Evaluation at x=5:", best_tree.evaluate(5))
