import numpy as np
import math
from typing import List
from scipy import stats
from simulation_methods.procedures import BernoulliPopulation


class SystemAnalyzer(object):
    comparison_methods = ['approximate']  # TODO: Add 'pooled' and 'paired'

    def __init__(self, results: List[List[float]]):
        if len(results) != 2:
            raise ValueError("Can only perform comparison on two independent systems")
        self.results = results

    def compare_systems(self, alpha: float, method: str) -> List[float]:
        if method not in self.comparison_methods:
            raise ValueError(f"Invalid comparison method. Use one of the following: {self.comparison_methods}")

        if method == 'approximate':
            return self._calculate_approximate_ci(alpha)

    def _calculate_approximate_ci(self, alpha: float) -> List[float]:
        num_samples_x = len(self.results[0])
        num_samples_y = len(self.results[1])

        sample_variance_x = np.var(self.results[0], ddof=1)
        sample_variance_y = np.var(self.results[1], ddof=1)

        approximate_dof = math.pow((sample_variance_x / num_samples_x) + (sample_variance_y / num_samples_y), 2) / (
                    (math.pow((sample_variance_x / num_samples_x), 2) / (num_samples_x + 1)) + (
                        math.pow((sample_variance_y / num_samples_y), 2) / (num_samples_y + 1))) - 2

        t = stats.t.ppf(1 - (alpha / 2), round(approximate_dof))
        lower_bound = np.mean(self.results[0]) - np.mean(self.results[1]) - t * np.sqrt(
            (sample_variance_x / num_samples_x) + (sample_variance_y / num_samples_y))
        upper_bound = np.mean(self.results[0]) - np.mean(self.results[1]) + t * np.sqrt(
            (sample_variance_x / num_samples_x) + (sample_variance_y / num_samples_y))

        return [lower_bound, upper_bound]


class System(object):

    def run_replication(self, iterations: int) -> float:
        raise NotImplementedError


class BernSystem(System):
    def __init__(self, population: BernoulliPopulation):
        self.population = population

    def run_replication(self, iterations: int) -> float:
        total = sum(self.population.generate_outcome() for i in range(iterations))
        result = total / iterations
        return result
