import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt

# this is all utility code to make the plots look nice without cluttering up the notebooks

def plot_nodes_and_edges(G, plt, fig, ax, seed):
    pos = nx.spring_layout(G, k=0.55, iterations=50, seed=seed)

    node_labels = nx.get_node_attributes(G, 'name')

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=10, edge_color='lightblue')
    nx.draw_networkx_nodes(G, pos, node_size=120, node_color='gray', ax=ax, alpha=0.4)

    edge_weights = nx.get_edge_attributes(G, 'weight')
    edge_weights_with_distance = { k: f'{round(v, 2)}\n{round(G.edges[k]["distance"])}m' for k, v in edge_weights.items() }

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights_with_distance, font_size=8, ax=ax)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    ax.set_xticks([])
    ax.set_yticks([])