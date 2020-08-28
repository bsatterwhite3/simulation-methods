import pytest
import numpy as np

from simulation_methods import procedures, methods


def test_approximate_ci():

    y1 = [10, 25, 5, 20, 15]
    y2 = [30, 15, 40, 10, 25]
    analyzer = methods.SystemAnalyzer([y1, y2])

    lower, upper = analyzer.compare_systems(.1, 'approximate')
    assert round(lower, 2) == -20.91
    assert round(upper, 2) == 2.91
