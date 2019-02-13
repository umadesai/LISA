import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from Graph_Wrapper import Graph, Name


def make_osm_graph(name, filepath):
    """
    Fetch OSM from name and pickle graph object to a file

    :param name: name of city to fetch osm from
    :param filepath: filepath to save the object
    :type filepath: string
    """
    G = Graph.from_bound(Name(name))
    G.save(filepath)


def load_osm_graph(filepath):
    """
    Unpickle the osm graph object

    :param filepath: filepath to the object
    :type filepath: string
    :rtype: Graph_Wrapper.Graph
    """
    pickle_in = open(filepath, "rb")
    return pickle.load(pickle_in)


def get_pink_node_coords(G):
    """
    Load pink node longitudes and latitudes as pandas dataframe

    :param G: Graph
    :G type: Graph_Wrapper.Graph
    :rtype: pandas dataframe
    """
    node_data = G.init_graph.nodes(data=True)
    coords = [(node[1]['x'], node[1]['y']) for node in node_data]
    return pd.DataFrame.from_records(coords, columns=['x', 'y'])


def get_signal_coords():
    """
    Load Signalized Intersections csv as pandas dataframe

    :rtype: pandas dataframe
    """
    return pd.read_csv("Signalized_Intersections_ACISA.csv",
                       usecols=["X", "Y"])


def plot_signalized_csv_over_osmnx(osmnx_node_df, signalized_csv_df):
    """
    Overlay scatterplots of two dataframes

    :param node_df: dataframe of osm node long/lat coordinates
    :node_df type: pandas dataframe
    :param signalized_df: dataframe of signalized intersection long/lat
     coordinates
    :signalized_df type: pandas dataframe
    :param title: title of plot
    :title type: string
    """
    ax = osmnx_node_df.plot(x="x", y="y", kind='scatter', color='b',
                            title='signalized csv over osmnx graph')
    signalized_csv_df.plot(x="X", y="Y", kind='scatter', color='g', ax=ax)
    plt.show()


def round_pair(pair):
    """
    Round the floats in pair to 4 decimals places

    :param pair: pair long/lat coordinates
    :pair type: tuple of floats
    :rtype: tuple of floats
    """
    if not isinstance(pair, tuple) or not isinstance(pair[0], float) \
            or not isinstance(pair[1], float):
        raise TypeError('Please provide a tuple of floats')
    return (round(pair[0], 4), round(pair[1], 4))


def round_signalized_intersections():
    """
    Round signalized intersection coordinates to 4 decimal places

    :rtype: set containing tuples of floats
    """
    signalized_intersections =  \
        get_signal_coords().to_records(index=False).tolist()
    return {round_pair(pair) for pair in signalized_intersections}


def update_graph(G):
    """
    Update nx graph with signalized attribute as boolean.

    :param G: Graph
    :G type: Graph_Wrapper.Graph
    """
    rounded_signals = round_signalized_intersections()
    signal_data = {}
    for pink_node in G.DiGraph.nodes(data=True):
        if round_pair((pink_node[1]['x'], pink_node[1]['y'])) in rounded_signals:
            k = pink_node[0]
            print("k:", k)
            signal_data[k] = {'signalized': True}
            # for yellow_node in G.node_map[k]:
            #     y = yellow_node[0]
            #     print("y: ", y)
            #     signal_data[y] = {'signalized': True}
    nx.set_node_attributes(G=G.DiGraph, values=signal_data)


def get_signalized_osmnx_nodes_as_df(G):
    node_data = G.DiGraph.nodes(data=True)
    signal_coords = []
    for node in node_data:
        if 'signalized' in node[1]:
            # print(node[1]['x'], node[1]['y'])
            signal_coords.append((node[1]['x'], node[1]['y']))
    return pd.DataFrame.from_records(signal_coords, columns=['x', 'y'])


def plot_signalized_node_overlay(signalized_osmnx_nodes_df, signalized_csv_df):
    ax = signalized_csv_df.plot(x="X", y="Y", kind='scatter', color='b',
                                title="signalized csv nodes in vlue, signalized osmnx nodes overlayed in green")
    signalized_osmnx_nodes_df.plot(x="x", y="y", kind='scatter', color='g', ax=ax)
    plt.show()


def plot_updated_osmnx_graph(signalized_osmnx_nodes_df, osmnx_node_df):
    ax = osmnx_node_df.plot(x="x", y="y", kind='scatter', color='b',
                            title="full osmnx graph in blue, signalized=true overlayed in green")
    signalized_osmnx_nodes_df.plot(x="x", y="y", kind='scatter', color='g', ax=ax)
    plt.show()


if __name__ == "__main__":

    # make_osm_graph("Washington DC",'/home/udesai/SCOPE/LISA/Code/dc.pickle')

    G = load_osm_graph('dc.pickle')
    osmnx_node_df = get_pink_node_coords(G)
    signalized_csv_df = get_signal_coords()
    plot_signalized_csv_over_osmnx(osmnx_node_df, signalized_csv_df)
    update_graph(G)
    signalized_osmnx_nodes_df = get_signalized_osmnx_nodes_as_df(G)
    plot_updated_osmnx_graph(signalized_osmnx_nodes_df, osmnx_node_df)
    plot_signalized_node_overlay(signalized_osmnx_nodes_df, signalized_csv_df)
