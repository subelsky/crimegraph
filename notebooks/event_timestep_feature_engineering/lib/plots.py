from matplotlib.font_manager import font_scalings
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt

def plot_subgraphs(graph, partition):
    # Get the unique partition values
    unique_partitions = set(partition.values())
    pos = nx.spring_layout(graph)
 
    for part in unique_partitions:
        plt.figure(figsize=(20, 20))

        # Create a subgraph for the current partition
        subgraph_nodes = [node for node in graph.nodes() if partition[node] == part]
        subgraph = graph.subgraph(subgraph_nodes)

        cmap = matplotlib.colormaps['viridis']

        # Draw the nodes and edges of the subgraph
        nx.draw_networkx_nodes(subgraph, pos, node_size=40, cmap=cmap, node_color=[part] * len(subgraph_nodes), alpha=0.2)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5)

        # Draw node labels (node 'name' attribute)
        node_labels = nx.get_node_attributes(subgraph, 'name')
        nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=6)

        edge_labels = nx.get_edge_attributes(subgraph, 'weight')
        edge_labels = { k: round(v, 2) for k, v in edge_labels.items() }

        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=6)

        # Show the plot for this subgraph
        plt.show()

def plot_subgraphs_on_map(graph, partition, gdf):
    for part in set(partition.values()):
        # Create a figure and axes object with a specified size
        fig, ax = plt.subplots(figsize=(20, 20))

        # Draw the street map as a light grey background on the specified axes
        gdf.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.5)

        # Get the nodes in the current partition
        subgraph_nodes = [node for node in graph.nodes() if partition[node] == part]
        subgraph = graph.subgraph(subgraph_nodes)

        # Get positions of the nodes
        pos = {node: (data['pos'][0], data['pos'][1]) for node, data in graph.nodes(data=True) if node in subgraph_nodes}

        nx.draw_networkx_nodes(subgraph, pos, node_size=40, node_color='blue', alpha=0.6, ax=ax)
        nx.draw_networkx_edges(subgraph, pos, edge_color='black', alpha=0.7, ax=ax)

        node_labels = {node: data['name'] for node, data in graph.nodes(data=True) if node in subgraph_nodes}

        edge_labels = nx.get_edge_attributes(subgraph, 'weight')
        edge_labels = { k: round(v, 2) for k, v in edge_labels.items() }

        nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=6, ax=ax)
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=6, ax=ax)

        plt.axis('off')
        plt.title(f'Subgraph {part}', fontsize=10)

        plt.show()

def plot_events_on_map(events_gdf):
    baltimore_shape = gpd.read_file('./../../data/city_geography/baltimore_no_water.shp')
    baltimore_shape = baltimore_shape.to_crs('EPSG:26985') # convert to NAD83, planar coordinates, in meters

    fig, ax = plt.subplots(figsize=(10, 10))

    events_gdf.plot(ax=ax, alpha=1.0, edgecolor='k', column='Type', legend=True)
    baltimore_shape.plot(ax=ax, color='lightgray', alpha=0.5)

    ax.set_title('Events in Baltimore City', fontsize=18, fontweight='bold')

    minLong, minLat, maxLong, maxLat = baltimore_shape.total_bounds

    ax.set_xlim(minLong, maxLong)
    ax.set_ylim(minLat, maxLat)

    ax.axis('off')
    plt.show()

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