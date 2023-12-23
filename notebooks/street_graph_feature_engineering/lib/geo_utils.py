from shapely.geometry import box
import geopandas as gpd

# downtown Baltimore bounding box
minLat, minLon = 39.29140363488898, -76.61435511031853
maxLat, maxLon = 39.28955544229775, -76.61075672041802

def make_bounding_box(minLat, minLon, maxLat, maxLon):
    '''Convert WGS 84 spherical coordinates to NAD 83 Maryland State Plane coordinates
       and return as a geoSeries containign a shapely box'''
    bbox = box(minLon, minLat, maxLon, maxLat)
    bbox_in_crs = gpd.GeoSeries([bbox], crs='EPSG:4326').to_crs('EPSG:26985').iloc[0]

    return bbox_in_crs

def filter_gdf(gdf, minLat=minLat, minLon=minLon, maxLat=maxLat, maxLon=maxLon):
    '''Filter a GeoDataFrame to only include geometries that intersect with the
       bounding box defined by the given coordinates'''
    bbox = make_bounding_box(minLat, minLon, maxLat, maxLon)
    within_bbox = gdf[gdf.intersects(bbox)]

    return within_bbox