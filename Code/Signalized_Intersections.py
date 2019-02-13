import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from Graph_Wrapper import Graph, Name


def make_osmnx_graph(name, filepath):
    """
    Fetch OSM from name and pickle graph object to a file

    :param name: name of city to fetch osm from
    :param filepath: filepath to save the object
    :type filepath: string
    """
    G = Graph.from_bound(Name(name))
    G.save(filepath)


def load_osmnx_graph(filepath):
    """
    Unpickle the osm graph object

    :param filepath: filepath to the object
    :type filepath: string
    :rtype: Graph_Wrapper.Graph
    """
    pickle_in = open(filepath, "rb")
    return pickle.load(pickle_in)


def get_osmnx_nodes(G):
    """
    Load pink node longitudes and latitudes as pandas dataframe

    :param G: Graph
    :G type: Graph_Wrapper.Graph
    :rtype: pandas dataframe
    """
    node_data = G.init_graph.nodes(data=True)
    coords = [(node[1]['x'], node[1]['y']) for node in node_data]
    return pd.DataFrame.from_records(coords, columns=['x', 'y'])


def get_signalized_csv_coords():
    """
    Load Signalized Intersections csv as pandas dataframe

    :rtype: pandas dataframe
    """
    return pd.read_csv("Signalized_Intersections_ACISA.csv",
                       usecols=["X", "Y"])


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
        get_signalized_csv_coords().to_records(index=False).tolist()
    return {round_pair(pair) for pair in signalized_intersections}


def update_graph(G):
    """
    Update nx graph with signalized attribute as boolean.

    :param G: Graph
    :G type: Graph_Wrapper.Graph
    :rtype: dictionary
    """
    rounded_signals = round_signalized_intersections()
    signal_data = {}
    for pink_node in G.DiGraph.nodes(data=True):
        if round_pair((pink_node[1]['x'], pink_node[1]['y'])) in rounded_signals:
            k = pink_node[0]
            signal_data[k] = {'signalized': True}
    nx.set_node_attributes(G=G.DiGraph, values=signal_data)
    return signal_data


def get_signalized_osmnx_nodes_as_df(G):
    node_data = G.DiGraph.nodes(data=True)
    coords = [(node[1]['x'], node[1]['y'])
              for node in node_data if 'signalized' in node[1]]
    return pd.DataFrame.from_records(coords, columns=['x', 'y'])


def plot_overlay(df1, df2, x1, y1, x2, y2, title):
    """
    Overlay scatterplots of two dataframes
    :param df1: pandas dataframe
    :param df2: pandas dataframe
    :param x1: x column name of df 1
    :param y1: y column name of df 1
    :param x2: x column name of df 2
    :param y2: y column name of df 2
    :param title: title of plot
    """
    ax = df1.plot(x=x1, y=y1, kind='scatter', color='b',
                  title=title)
    df2.plot(x=x2, y=y2, kind='scatter', color='g',
             ax=ax)
    plt.show()


if __name__ == "__main__":

    # make_osmnx_graph("Washington DC",'/home/udesai/SCOPE/LISA/Code/dc.pickle')

    G = load_osmnx_graph('dc.pickle')
    osmnx_node_df = get_osmnx_nodes(G)
    signalized_csv_df = get_signalized_csv_coords()
    # plot signalized csv nodes over osmnx nodes
    plot_overlay(osmnx_node_df, signalized_csv_df, 'x', 'y',
                 'X', 'Y', 'signalized csv over osmnx graph')
    # update osmnx graph attributes for signalization
    update_graph(G)
    # plot signalized osmnx graph nodes over full osmnx graph
    signalized_osmnx_nodes_df = get_signalized_osmnx_nodes_as_df(G)
    plot_overlay(osmnx_node_df, signalized_osmnx_nodes_df, 'x', 'y', 'x', 'y',
                 "full osmnx graph in blue, signalized=true overlayed in green")
    # plot signalized osmnx nodes over signalized csv nodes
    plot_overlay(signalized_csv_df, signalized_osmnx_nodes_df, 'X', 'Y', 'x', 'y',
                 "signalized csv nodes in blue, signalized osmnx nodes overlayed in green")
