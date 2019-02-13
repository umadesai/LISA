from Signalized_Intersections import load_osm_graph, update_graph, \
    get_signal_coords, trunc, get_pink_node_coords
import pytest


def test_trunc():
    assert trunc((70.001201, 38.9011924)) == (70.0012, 38.9012)


@pytest.mark.parametrize("long, lat", [
    ((70, 39)),
    ((70, 31.2)),
    ((70.1, 39)),
])
def test_trunc_exception(long, lat):
    with pytest.raises(TypeError) as e:
        trunc((long, lat))
    assert str(e.value) == 'Please provide a tuple of floats'


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
