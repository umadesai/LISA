from Graph_Wrapper import KDTreeWrapper
from Signalized_Intersections import load_osmnx_graph
import geopandas as gpd
import requests
import os

URL = "https://api.mapbox.com/matching/v5/mapbox/cycling/"
# export MY_API_KEY=this_is_my_api_key
MY_API_KEY = os.environ.get('MY_API_KEY', None)


def get_ride_report_paths():
    """
    Extract the x,y coordinates for each ride report gps signal, returned as a
    list of lists for each path
    """
    gdf = gpd.read_file('RideReportRoutes.geojson')
    linestrings = gdf.geometry
    return [[coord for coord in line.coords] for line in linestrings]


def get_map_box_match(path):
    """
    Make a call to MapBox's Map Matching API to find the most likely osm path
    from the inputed ride report path
    """
    waypoints = ""
    for tup in path:
        waypoints += str(tup[0]) + "," + str(tup[1]) + ";"
    endpoint = URL + waypoints[:-1] + "?access_token=" + MY_API_KEY
    resp = requests.get(endpoint).json()
    return [point['location'] for point in resp['tracepoints']]


def get_closest_osmnx_path(map_box_path, kd_tree):
    """
    For each long/lat value in map_box_path, return the closest node in the
    osmnx graph
    """
    path = []
    for tup in map_box_path:
        closest_node, d = kd.query_min_dist_nodes(tup)
        print("distance between matched nodes: ", d)
        path.append(closest_node)
    return path


if __name__ == "__main__":

    G = load_osmnx_graph('dc.pickle')
    kd = KDTreeWrapper(G.DiGraph)
    paths = get_ride_report_paths()
    match = get_map_box_match(paths[0])
    osmnx_path = get_closest_osmnx_path(match, kd)
