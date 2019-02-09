from Graph_Wrapper import Graph, Name
from Signalized_Intersections import *
import folium
import random
from sklearn.neighbors import KDTree
import json

def getRandomPath(node_map, path_len):
    """
        Generates a path of length path_len from a random starting point

        :param node_map: dict
        :rtype: list of node IDs (int)
    """
    start = random.choice(list(node_map.keys()))
    route = [start]
    while (path_len > 1):
        neighbors = list(node_map[start])
        random_neighbor = random.choice(neighbors)
        route.append(random_neighbor)
        start = random_neighbor
        path_len -= 1
    return route

def displayRoute(route):
    """
        Plots the routes onto a geometric version of the boundary using folium and OSMnx
    """
    # TODO: use ox.plot_graph_route to plot it in an actual plot
    # reference: https://medium.com/@bobhaffner/osmnx-intro-and-routing-1fd744ba23d8
    pass

if __name__ == "__main__":
    # make_osm_graph("Washington DC",'DIRECTORY/LISA/Code/dc.pickle')
    DC_graph = load_osm_graph('dc.pickle')
    node_map = DC_graph.node_map
    print(getRandomPath(node_map, 3))
    # ISSUES: Graph_Wrapper update_nodes_xy mentions "unit_vec" before assignment in else statement
    # Moving unit_vec up above if statements makes limited keys in the node_map