from map_matching import get_candidates, cyclist_distance, calculate_dist
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


# def test_get_candidates(kd):
#     assert get_candidates((23423, 23432), kd) == (23423, 2342), (89343, 345934)
#
#
# def test_cyclist_distance(G):
#     assert cyclist_distance(G, 234, 541) == 9340
#
#
# def test_calculate_dist(G):
#     assert calculate_dist(G, 2392, 24924) == 823
