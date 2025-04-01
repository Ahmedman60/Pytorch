from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Add this before importing pyplot
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

# Draw the graph
plt.figure(figsize=(6, 4))
pos = nx.spring_layout(G)  # Layout for better visualization
nx.draw(G, pos, with_labels=True, node_color='lightblue',
        edge_color='gray', node_size=2000, font_size=12, font_weight='bold')

# Show the graph
plt.savefig("graph.png")  # You already have this
print("Graph saved as 'graph.png'")

# ----------------------------------------------------------------
# Initialize a queue

# def bfs(graph, node):  # function for BFS
#     queue = deque()
#     visited = set(node)  # List of

#     queue.append(node)
#     while queue:          # Creating loop to visit each node
#         m = queue.popleft()
#         print(m, end=" ")

#         for neighbour in graph[m]:
#             if neighbour not in visited:
#                 visited.append(neighbour)
#                 queue.append(neighbour)
