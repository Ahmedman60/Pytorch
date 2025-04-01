import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Define the graph as an adjacency list
graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': []
}

# Create a directed graph using NetworkX
G = nx.DiGraph()

# Add edges to the graph
for node, neighbors in graph.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Create a new figure and axis explicitly
fig, ax = plt.subplots(figsize=(6, 4))

# Draw the graph on the specified axis
pos = nx.spring_layout(G)  # Layout for better visualization
nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
        edge_color='gray', node_size=2000, font_size=12, font_weight='bold')

# Set the title
ax.set_title("Graph Visualization")

# Save and display
plt.savefig("graph.png")
plt.show()


# Define the graph as an adjacency list
graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': []
}


queue = deque()


def BFS(graph, root):
    visited = set(root)
    queue.append(root)
    while queue:
        node = queue.popleft()
        print(node)
        for current_node in graph[node]:
            if current_node not in visited:
                visited.add(current_node)

                queue.append(current_node)
            # queue.extend(graph[current_node])


BFS(graph, '5')
