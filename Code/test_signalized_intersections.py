from Signalized_Intersections import load_osm_graph, update_graph, \
    get_signal_coords, trunc, get_pink_node_coords
import pytest

# the @pytest.mark.parametrize decorator allows us to test different
# combinations of values in one function


@pytest.mark.parametrize("long, lat, expected_long, expected_lat", [
    ((70.001201, 38.9011924, 70.0012, 38.9012)),
    ((69.899980, -39.0131429, 69.9000, -39.0131)),
    ((70, 39, 70, 39))
])
def test_trunc(long, lat, expected_long, expected_lat):
    assert trunc((long, lat)) == (expected_long, expected_lat)

# the @pytest.fixture decorator helps us avoid repition when we set up helper
# code like loading an osmnx graph or initializing a class


@pytest.fixture
def G():
    '''
    Returns osmnx graph of DC
    '''
    return load_osm_graph('dc.pickle')


def test_get_signal_coords():
    assert len(get_signal_coords()) > 1400


def test_get_pink_node_coords(G):
    assert len(get_pink_node_coords(G)) > 3800


def test_update_graph(G):
    update_count = update_graph(G)
    assert update_count > 1400
