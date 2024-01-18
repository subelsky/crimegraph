from matplotlib.font_manager import font_scalings
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

def plot_event_counts(timesteps, node_index, event_type, num_days=100):
    # Group by date to get the sum for one particular node identified by a NodeIndex value
    daily_totals = timesteps.xs(node_index, level='NodeIndex').groupby(level='Date').sum()
    daily_totals = daily_totals.tail(num_days)

    plt.figure(figsize=(20, 5))

    plt.vlines(daily_totals.index, ymin=0, ymax=daily_totals[event_type], colors='black', linewidth=1, label=f'{event_type} daily event count')
    plt.plot(daily_totals.index, daily_totals[f'{event_type}Smoothed'], color='red', label=f'Smoothed {event_type} count', linewidth=1)

    plt.ylim(bottom=0, top=daily_totals[event_type].max() * 1.1)
    plt.xlim(daily_totals.index.min(), daily_totals.index.max())

    plt.xlabel('Day')
    plt.ylabel('Event count')
    plt.title(f'{event_type} events for NodeIndex {node_index}')
    plt.legend()
    plt.tight_layout()

    plt.show

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