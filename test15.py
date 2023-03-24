
import networkx as nx

# Create a graph
G = nx.DiGraph()
G.add_edges_from([(1, 2), (2, 3), (2, 4), (4, 5)])

# Find the depth of node 5
depth = nx.shortest_path_length(G, source=5, target=1)

print(depth)  # Output: 3

