import numpy as np

from typing import List


class BrownianMotion(object):

    def __init__(self, mean=0, stdev=1, seed=None):
        self.mean = mean
        self.stdev = stdev

        self.wiener_vector = []
        if seed:
            np.random.seed(seed)

    @property
    def result(self) -> float:
        if len(self.wiener_vector) == 0:
            return np.NaN
        return self.wiener_vector[-1]

    @property
    def history(self) -> List[float]:
        return self.wiener_vector

    def _step(self, current):
        return current + np.random.normal(self.mean, self.stdev)

    def run_simulation(self, num_steps=1000, initial_value=0):
        current = initial_value
        self.wiener_vector = [current]
        for i in range(0, num_steps):
            current = self._step(current)
            self.wiener_vector.append(current)
