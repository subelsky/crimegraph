import numpy as np
from sklearn.neighbors import BallTree

def associate_events_with_nodes(events_gdf, nodes_gdf):
    """Associates each event in a GeoDataFrame with the nearest node in the street network"""

    event_coords = np.array(list(events_gdf.geometry.apply(lambda geom: (geom.x, geom.y))))
    node_coords = np.array(list(nodes_gdf.geometry.apply(lambda geom: (geom.x, geom.y))))

    # Create a BallTree for efficient nearest neighbor queries
    tree = BallTree(node_coords, leaf_size=2)

    # Query the tree for the nearest node for each event
    distances, indices = tree.query(event_coords, k=1)
    nearest_node_id = nodes_gdf.iloc[indices.flatten()].index

    # Add the nearest node ID as a column in the events_gdf
    events_gdf['NodeIndex'] = nearest_node_id

    return events_gdf