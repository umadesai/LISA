from Signalized_Intersections import load_osmnx_graph, update_graph, \
    get_signalized_csv_coords, round_pair, get_osmnx_nodes
import pytest


def test_round_pair():
    assert round_pair((70.001201, 38.9011924)) == (70.0012, 38.9012)


@pytest.mark.parametrize("long, lat", [
    ((70, 39)),
    ((70, 31.2)),
    ((70.1, 39)),
])
def test_round_pair_exception(long, lat):
    with pytest.raises(TypeError) as e:
        round_pair((long, lat))
    assert str(e.value) == 'Please provide a tuple of floats'


@pytest.fixture
def G():
    '''
    Returns osmnx graph of DC
    '''
    return load_osmnx_graph('dc.pickle')


def test_get_signalized_csv_coords():
    assert len(get_signalized_csv_coords()) > 1400


def test_get_osmnx_nodes(G):
    assert len(get_osmnx_nodes(G)) > 3800


def test_update_graph(G):
    update_dictionary = update_graph(G)
    assert len(update_dictionary) > 1400
