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


def overlay_dfs(node_df, signalized_df, title):
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
    ax = node_df.plot(x="x", y="y", kind='scatter', color='b',
                      title=title)
    signalized_df.plot(x="X", y="Y", kind='scatter', color='g', ax=ax)
    plt.show()


def trunc(pair):
    """
    Round the floats in pair to 4 decimals places

    :param pair: pair long/lat coordinates
    :pair type: tuple of floats
    :rtype: tuple of floats
    """
    return (round(pair[0], 4), round(pair[1], 4))


def trunc_signalized_intersections():
    """
    Round signalized intersection coordinates to 4 decimal places

    :rtype: set containing tuples of floats
    """
    signalized_intersections =  \
        get_signal_coords().to_records(index=False).tolist()
    return {trunc(pair) for pair in signalized_intersections}


def update_graph(G):
    """
    Update nx graph with signalized attribute as boolean. 

    :param G: Graph
    :G type: Graph_Wrapper.Graph
    :rtype: int
    """
    trunc_signals = trunc_signalized_intersections()
    count = 0
    signal_data = {}
    for pink_node in G.DiGraph.nodes(data=True):
        if trunc((pink_node[1]['x'], pink_node[1]['y'])) in trunc_signals:
            k = pink_node[0]
            signal_data[k] = {'signalized': True}
            # print('Found a signalized intersection')
            count += 1
            # for yellow_node in G.node_map[pink_node]:
            #     signal_data[yellow_node[0]] = {'signalized': True}
    nx.set_node_attributes(G=G.DiGraph, values=signal_data)
    return count


if __name__ == "__main__":

    # make_osm_graph("Washington DC",'/home/udesai/SCOPE/LISA/Code/dc.pickle')
    
    G = load_osm_graph('dc.pickle')
    node_df = get_pink_node_coords(G)
    signalized_df = get_signal_coords()
    overlay_dfs(node_df, signalized_df, "Signalized Intersections in DC")
    update_graph(G)
