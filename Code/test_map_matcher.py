from Graph_Wrapper import KDTreeWrapper
from Signalized_Intersections import load_osmnx_graph
import pytest


@pytest.fixture
def G():
    '''
    Returns osmnx graph of DC
    '''
    return load_osmnx_graph('dc.pickle')


@pytest.fixture
def kd(G):
    '''
    Returns kd tree
    '''
    return KDTreeWrapper(G.DiGraph)
