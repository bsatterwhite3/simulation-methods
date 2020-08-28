import pytest
import numpy as np

from simulation_methods import models


class TestBrownianMotion(object):

    def test_brownian_step(self, mocker):
        mocker.patch('simulation_methods.models.np.random.normal', return_value=0.5)

        bm = models.BrownianMotion()
        result = bm._step(3)
        assert result == 3.5

    def test_brownian_simulation(self, mocker):
        mocker.patch('simulation_methods.models.np.random.normal', return_value=0.5)

        bm = models.BrownianMotion()
        steps = 5
        bm.run_simulation(steps, 0)

        expected_history = [0, 0.5, 1, 1.5, 2, 2.5]
        expected_result = 2.5

        assert bm.history == expected_history
        assert bm.result == expected_result
