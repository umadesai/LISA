from map_matching import cyclist_distance
import pytest


def test_emission_probability():
    assert cyclist_distance(0, 0, 0) == 200
