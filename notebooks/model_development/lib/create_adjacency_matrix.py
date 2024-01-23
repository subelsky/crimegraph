import networkx as nx
import numpy as np

def create_adjacency_matrix(G):
    n = G.number_of_nodes()
    weighted_adj_matrix = np.zeros((n, n))

    # Iterate through the graph edges to fill in the matrix
    for edge in G.edges(data=True):
        i, j, data = edge
        weight = data['weight'] # computed in the step 1 notebook

        # Weights are equal in both directions, because graph is undirected
        weighted_adj_matrix[i, j] = weight
        weighted_adj_matrix[j, i] = weight

    return weighted_adj_matrix
