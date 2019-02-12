from Signalized_Intersections import load_osm_graph, update_graph, \
    get_signal_coords, trunc, get_pink_node_coords


def test_get_signal_coords():
    assert len(get_signal_coords()) > 1400


def test_get_pink_node_coords():
    G = load_osm_graph('dc.pickle')
    assert len(get_pink_node_coords(G)) > 3800


def test_trunc():
    test_pair = ((70.001201, 38.9011924))
    assert trunc(test_pair) == (70.0012, 38.9012)


def test_update_graph():
    G = load_osm_graph('dc.pickle')
    update_count = update_graph(G)
    assert update_count > 1400


# python -m pytest Test_Signalized_Intersections.py
