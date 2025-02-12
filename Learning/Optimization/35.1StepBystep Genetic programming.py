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


def generate_random_tree(depth=3):
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


parent1 = generate_random_tree(depth=2)
parent2 = generate_random_tree(depth=2)

print("Parent 1:")
parent1.print()
print("\nParent 2:")
parent2.print()

child1, child2 = crossover(parent1, parent2)

print("\nChild 1 (After Crossover):")
child1.print()
print("\nChild 2 (After Crossover):")
child2.print()
