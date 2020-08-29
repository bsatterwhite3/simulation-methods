import numpy as np

from typing import List
from collections import namedtuple


class BrownianMotion(object):

    BrownianOutcome = namedtuple('BrownianOutcome', ['history_vector', 'result'])

    def __init__(self, mean=0, stdev=1, seed=None):
        self.mean = mean
        self.stdev = stdev

        if seed:
            np.random.seed(seed)

    def _step(self, current):
        return current + np.random.normal(self.mean, self.stdev)

    def run_simulation(self, num_steps=1000, initial_value=0) -> BrownianOutcome:
        """Runs a brownian motion simulation based on the initial mean and standard deviation provided."""
        current = initial_value
        history_vector = [current]
        for i in range(0, num_steps):
            current = self._step(current)
            history_vector.append(current)

        return self.BrownianOutcome(history_vector=history_vector, result=current)
