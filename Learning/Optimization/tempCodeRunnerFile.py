
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