import networkx as nx
import numpy as np
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
from sklearn.neighbors import BallTree

# this value was chosen by trial-and-error; the GLDnet paper used a value of 5.
# At training time, we can recompute the edge weights with different alpha values,
# which means we can find a more optimal value through hyperparameter tuning.
DEFAULT_ALPHA = 750

def create_weight_calculator(alpha):
    """
    Creates a function that calculates the weight of an edge between two intersection street segments
    using a Gaussian kernel function

    Parameters
    ----------
    alpha : float
        The standard deviation of the Gaussian kernel function. This is a hyperparameter that
        controls the rate of decay of the weights as the distance between two intersecting street
        segments increases. A higher value of alpha will result in a slower decay rate, meaning
        that the weights will decrease more slowly as the distance between two intersection street
        increases
    """
    two_alpha_squared = (2 * alpha**2)

    def calculate_weight(distance):
        weight = np.exp(-(distance ** 2) / two_alpha_squared)

        return weight

    return calculate_weight

def create_network_graph(gdf, alpha=DEFAULT_ALPHA):
    """Creates a networkx graph from a GeoDataFrame of street segments"""

    G = nx.Graph()
    weight_calculator = create_weight_calculator(alpha)

    for index, row in gdf.iterrows():
        # Use the index of the GeoDataFrame as the node identifier
        G.add_node(index, name=row['Name'], position=row['geometry'].centroid.coords[0])

    # Create a dictionary to keep track of the nodes connected by an intersection
    intersections = {}

    # Find intersections by iterating through each row of the GeoDataFrame;
    # there may be a more efficient way to do this
    for idx1, seg1 in gdf.iterrows():
        for idx2, seg2 in gdf.iterrows():
            if idx1 != idx2:
                if seg1['geometry'].intersects(seg2['geometry']):
                    intersection = seg1['geometry'].intersection(
                        seg2['geometry'])

                    if isinstance(intersection, Point):
                        # Single point of intersection
                        if (idx1, idx2) not in intersections and (idx2, idx1) not in intersections:
                            intersections[(idx1, idx2)] = intersection

                    elif isinstance(intersection, MultiPoint):
                        # Multiple points of intersection; take the one closest to the midpoint of seg1
                        nearest_intersection = nearest_points(
                            seg1['geometry'].centroid, intersection)[1]

                        if (idx1, idx2) not in intersections and (idx2, idx1) not in intersections:
                            intersections[(idx1, idx2)] = nearest_intersection

    for (idx1, idx2), intersection_point in intersections.items():
        # Network distance is the distance along the street to the intersection point
        distance1 = gdf.at[idx1, 'geometry'].project(intersection_point)
        distance2 = gdf.at[idx2, 'geometry'].project(intersection_point)

        # Edge weight is the average of the distances from each node to the intersection point
        distance = (distance1 + distance2) / 2
        weight = weight_calculator(distance)

        # also storing the distance value so that the weights can be re-computed during training,
        # for different values of alpha
        G.add_edge(idx1, idx2, distance=distance, weight=weight)

    return G
