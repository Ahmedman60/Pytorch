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


generate_random_tree().print()


'''
+
 - 
   *
     1
     x
   /
    x
    3
 *
    *
     1
     x
   /
    x
    3
'''
