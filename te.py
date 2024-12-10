import random


# dividing groups of random names into groups

names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve',
         'Frank', 'Grace', 'Harry', 'Ivy', 'Jack']
groups = 4

random.shuffle(names)

# dividing names into groups of 5
groups = [[] for _ in range(groups)]

for i, name in enumerate(names):
    groups[i % len(groups)].append(name)

# printing groups

for i, group in enumerate(groups):
    print(f"Group {i}: {', '.join(group)}")
