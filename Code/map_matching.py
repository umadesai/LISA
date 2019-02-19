import math
import numpy as np
import networkx as nx
import geopandas as gpd
from Graph_Wrapper import KDTreeWrapper
from Signalized_Intersections import load_osmnx_graph

"""
Hidden Markov Model
Given a sequence of GPS signals (observations), find the most probable
sequence of road segments (hidden states)
"""

SIGMA_Z = 4.07
BETA = 3


def get_ride_report_paths():
    """
    Extract the x,y coordinates for each gps signal, returned as a list of
    lists for each path
    """
    gdf = gpd.read_file('RideReportRoutes.geojson')
    linestrings = gdf.geometry
    return [[coord for coord in line.coords] for line in linestrings]


def first_pass(paths, kd_tree):
    """
    For each gps signal, return the closest node
    """
    return [[kd.query_min_dist_nodes(node) for node in path] for path in paths]


def get_candidates(gps_point, kd):
    """
    Grab 2 closest nodes to gps point
    """
    nodes, distances = kd.query_min_dist_nodes(gps_point, k=2)
    return (nodes[0], distances[0]), (nodes[1], distances[1])


def calculate_dist(G, p0, p1):
    """
    Find distance between two nodes
    """
    node_data = G.init_graph.nodes(data=True)
    p0 = (node_data[p0]['x'], node_data[p0]['y'])
    p1 = (node_data[p1]['x'], node_data[p1]['y'])
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def cyclist_distance(G, source, target):
    """
    Find cyclist distance between two nodes
    """
    # path = nx.algorithms.shortest_paths.astar.astar_path(G, source, target)
    # print(path)
    return 200


def emission_probability(node):
    """
    For each street segment the probability of a given GPS point matching the
    street is inverse-propositional to its distance.
    """
    # A gaussian distribution
    d = node[1]
    c = 1 / (SIGMA_Z * np.sqrt(2 * np.pi))
    return c * np.exp(d**2)


def transition_probability(G, node1, node2):
    """
    Measure the probability to transition from one street segment to any other
    street segment. Newson and Krumm found that a large difference between the
    driving distance of two street segments and the straight line distance
    between the corresponding GPS samples point to an unlikely transition.
    """
    # An empirical distribution
    c = 1 / BETA
    delta = abs(cyclist_distance(G, node1, node2) -
                calculate_dist(G, node1, node2))
    return c * np.exp(-delta)


def path_probability(G, path):
    """
    A path is a list of (node, gps_signal)
    Calculate joint probability of each point in path, returns product
    """
    start = path[0]
    p = emission_probability(start)
    for i in range(1, len(path)):
        p *= transition_probability(G, path[i-1][0], path[i][0]) * \
            emission_probability(path[i][0], path[i][1])
    return p


def maximum_path_prob(G, gps_signals, kd):
    """
    Find the path from a list of gps signals that maximizes path probability
    """
    paths = all_paths(gps_signals, [], kd)
    max_prob = 0
    max_path = 0
    for path in paths:
        curr = path_probability(G, path)
        if curr > max_prob:
            max_prob = path_probability(G, path)
            max_path = path
    return max_path


def all_paths(gps_signals, paths, kd):
    """
    Generate all possible paths from a list of gps signals recursively
    """
    c1, c2 = get_candidates(gps_signals[0], kd)
    if len(gps_signals) < 2:
        if len(paths) < 1:
            return [[c1], [c2]]
        else:
            new_paths = []
            for path in paths:
                path1 = path + [c1]
                path2 = path + [c2]
                new_paths.append(path1)
                new_paths.append(path2)
            return new_paths
    else:
        if len(paths) < 1:
            return all_paths(gps_signals[1:], [[c1], [c2]], kd)
        new_paths = []
        for path in paths:
            path1 = path + [c1]
            path2 = path + [c2]
            new_paths.append(path1)
            new_paths.append(path2)
        return all_paths(gps_signals[1:], new_paths, kd)


def viterbi_algorithm():
    """
    An algorithm to identify the most probable sequence of states based on the
    probabilities of observation and transitions.

    Instead of using brute force to compute every possible path's probability,
    the viterbi algorithm computes incremental path probabilities, working from
    left to right. This works because HMMs satisfy the Markov prooperty, that
    the state at timestep t+1 depends only on timestep t.
    """
    pass


if __name__ == "__main__":

    G = load_osmnx_graph('dc.pickle')
    kd = KDTreeWrapper(G.DiGraph)
    # rr_paths = get_ride_report_paths()
    # matched_paths = first_pass(rr_paths, kd)
    gps_signals = [(2342, 2324), (3453, 29524)]
    p = maximum_path_prob(G, gps_signals, kd)
