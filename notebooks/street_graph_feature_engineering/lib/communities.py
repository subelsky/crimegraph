from community import community_louvain
import networkx as nx

# Unused code for an experiment to try using the Louvain algorithm to find
# smaller subgraphs within the street graph
def create_subgraphs(G, resolution=1.0):
    G_copy = G.copy()

    isolated_nodes = list(nx.isolates(G_copy))
    G.remove_nodes_from(isolated_nodes)

    partition = community_louvain.best_partition(G_copy, resolution=resolution)

    return partition