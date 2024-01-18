import numpy as np
from sklearn.neighbors import BallTree
import pandas as pd

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

# Set the smoothing factor in the GLDNet paper, subject to Hyperparameter Optimization
DEFAULT_SMOOTHING_FACTOR = 0.5

# This was the initial attempt, which did not account for event sparseness
def perform_exponential_smoothing(df, smoothing_factor=DEFAULT_SMOOTHING_FACTOR):
    df['ArrestSmoothed'] = df.groupby(level='NodeIndex')['Arrest'].transform(
        lambda x: x.ewm(alpha=smoothing_factor, adjust=False).mean()
    )

    df['PropertySmoothed'] = df.groupby(level='NodeIndex')['Property'].transform(
        lambda x: x.ewm(alpha=smoothing_factor, adjust=False).mean()
    )

    df['ViolentSmoothed'] = df.groupby(level='NodeIndex')['Violent'].transform(
        lambda x: x.ewm(alpha=smoothing_factor, adjust=False).mean()
    )

    return df

def interpolate_and_smooth(df, smoothing_factor=DEFAULT_SMOOTHING_FACTOR):
    """Interpolates missing event dates and performs exponential smoothing on each event type column for each node"""

    # Create a date range for the complete time period
    date_range = pd.date_range(start=df.index.get_level_values('Date').min(), end=df.index.get_level_values('Date').max())

    # Initialize an empty DataFrame to store the results
    full_df = pd.DataFrame()

    for node in df.index.get_level_values('NodeIndex').unique():
        # Get the subset of the DataFrame for the current node
        node_df = df.xs(node, level='NodeIndex')
        
        # Reindex the node's DataFrame to include the complete date range, filling missing values with 0
        node_df_reindexed = node_df.reindex(date_range).fillna(0)
        
        # Add the NodeIndex back to the DataFrame
        node_df_reindexed['NodeIndex'] = node
        
        # Apply exponential smoothing to each event type column
        for event_type in ['Arrest', 'Property', 'Violent']:
            node_df_reindexed[f'{event_type}Smoothed'] = node_df_reindexed[event_type].ewm(alpha=smoothing_factor, adjust=False).mean()
        
        # Concatenate this node's DataFrame to the full DataFrame
        full_df = pd.concat([full_df, node_df_reindexed])

    # Reset the index after concatenation
    full_df.reset_index(inplace=True)

    # Rename 'index' column to 'Date' to reflect the actual data
    full_df.rename(columns={'index': 'Date'}, inplace=True)

    # Set the index of the full DataFrame to include 'NodeIndex' and 'Date' again
    full_df.set_index(['Date', 'NodeIndex'], inplace=True)

    # full_df now contains the original data with missing dates filled with zero counts,
    # and the smoothed values for each event type.

    return full_df
