import random
import logging
import numpy as np

from typing import List, Dict
from collections import namedtuple
from scipy.stats import bernoulli

logger = logging.getLogger(__name__)


class Population(object):
    pass


class BernoulliPopulation(Population):

    def generate_outcome(self):
        raise NotImplementedError


class GenericBernoulliPopulation(BernoulliPopulation):
    def __init__(self, prob: float):
        self.prob = prob

    def generate_outcome(self) -> int:
        outcome = bernoulli.rvs(self.prob, size=1)
        return int(outcome)


class Selection(object):

    def __init__(self, populations):
        self.populations = populations

    def select_best_population(self):
        raise NotImplementedError


class BernoulliBechhoferKulkarni(Selection):
    """This class is an implementation of the sequential procedure for selecting the best bernoulli population
        as described by Bechhofer and Kulkarni (1982): https://apps.dtic.mil/sti/citations/ADA115219

        This procedure samples a group of Bernoulli populations until it reaches the max number of samples requested,
        or until one population is clearly better than the others.
    """
    ObservationCounter = namedtuple('ObservationCounter', ['num_observations', 'successes', 'failures'])

    def __init__(self, populations: List[BernoulliPopulation], max_samples: int):
        super().__init__(populations)
        self.max_samples = max_samples
        self.observation_map = self._initialize_procedure()
        self.observation_history = []

    def select_best_population(self) -> int:
        self._run_procedure()
        return self._get_index_of_highest_success(include_terminated_populations=True)

    def _run_procedure(self):
        """This function implements the BernoulliBechhoferKulkarni selection procedure.

            Steps:
                1. Select a population to take the next observation from by finding the one with the smallest
                number of failures
                2. If number of failures is tied, use the population with the highest number of successes instead
                3. Generate an outcome from the population and update the existing observations accordingly
                4. Evaluate whether termination criteria has been satisfied
                5. Repeat steps 1-4
        """
        logger.info(f"Starting BernoulliBechhoferKulkarni selection procedure with {len(self.populations)} populations")
        reached_termination = False
        while not reached_termination:
            population_index = self._get_next_population()
            population = self.populations[population_index]

            observation = population.generate_outcome()
            self._update_observation_map(population_index, observation)
            reached_termination = self._termination_criteria_satisfied()

        logger.info(f"Termination criteria reached. Observation counts: {self.observation_map}")

    def _initialize_procedure(self) -> Dict[int, ObservationCounter]:
        observation_map = {}
        for i in range(len(self.populations)):
            observation_map[i] = self.ObservationCounter(
                num_observations=0,
                successes=0,
                failures=0
            )
        return observation_map

    def _get_index_of_lowest_failure(self) -> int:
        min_index = -1
        min_failures = np.Inf

        for i in self.observation_map.keys():
            observation_counter = self.observation_map[i]
            if observation_counter.num_observations < self.max_samples:
                if observation_counter.failures == min_failures:
                    return -1

                if observation_counter.failures < min_failures:
                    min_failures = observation_counter.failures
                    min_index = i

        return min_index

    def _get_index_of_highest_success(self, include_terminated_populations=False) -> int:
        max_index = []
        max_successes = 0

        for i in self.observation_map.keys():
            observation_counter = self.observation_map[i]
            if observation_counter.num_observations < self.max_samples or include_terminated_populations:
                if observation_counter.successes == max_successes:
                    max_index.append(i)

                if observation_counter.successes > max_successes:
                    max_successes = observation_counter.successes
                    max_index = [i]
        return random.choice(max_index)

    def _get_next_population(self):
        """This function gets the index for the next population to take observations from.

            If the number of failures is tied across populations, we take the highest success.
        """
        failure_index = self._get_index_of_lowest_failure()
        return failure_index if failure_index != -1 else self._get_index_of_highest_success()

    def _update_observation_map(self, index: int, observation: int):
        previous_observation = self.observation_map[index]
        if observation == 1:
            observation_count = self.ObservationCounter(
                num_observations=previous_observation.num_observations + 1,
                failures=previous_observation.failures,
                successes=previous_observation.successes + 1
            )
        elif observation == 0:
            observation_count = self.ObservationCounter(
                num_observations=previous_observation.num_observations + 1,
                failures=previous_observation.failures + 1,
                successes=previous_observation.successes
            )
        else:
            raise ValueError(f"Encountered invalid observation {observation}")
        self.observation_map[index] = observation_count
        self.observation_history.append((index, observation_count))

    def _termination_criteria_satisfied(self) -> bool:
        """This function evaluates whether the procedure has satisfied the BernoulliBechhoferKulkarni evaluation criteria.

            In the procedure, sampling ends when Y_im >= Y_jm + n - n_jm for all j != i (1 <= i, j <= k)
            In simple English, this states that we should halt when no other populations can 'catch up' to the one
            with the most successes when factoring in the maximum number of samples we want to take per population.
        """
        max_index = self._get_index_of_highest_success(include_terminated_populations=True)
        max_successes = self.observation_map[max_index].successes
        if all(observation_counter.num_observations == self.max_samples
               for observation_counter in self.observation_map.values()):
            return True

        if all(max_successes >=
               self.observation_map[i].successes + self.max_samples - self.observation_map[i].num_observations
               for i in self.observation_map.keys() if i != max_index):
            return True

        return False


class BernoulliBechhoferKieferSobel(Selection):
    """This class is an implementation of the sequential procedure for selecting the best bernoulli population
        as described by Bechhofer, Kiefer, and Sobel in the book:
        https://books.google.com/books/about/Sequential_identification_and_ranking_pr.html?id=ixC3rvnHK5sC

        This procedure uses an iterative approach with an indifference zone and a probability of correct selection P(CS)
         to determine which population is best.
    """
    def __init__(self, populations, indifference, desired_probability):
        super().__init__(populations)
        self._validate_procedure_inputs(indifference, desired_probability)
        self.indifference = indifference
        self.desired_probability = desired_probability
        self.counts = np.zeros(len(populations))

    def select_best_population(self) -> int:
        self._run_procedure()
        return int(np.argmax(self.counts))

    def _validate_procedure_inputs(self, indifference, desired_probability):
        if indifference < 1:
            raise ValueError(f"Procedure only terminates when indifference is greater than 1. Provided: {indifference}")

        if desired_probability >= 1 or desired_probability <= 0:
            raise ValueError(f"Probability must be between 0 and 1. Provided: {desired_probability}")

    def _run_procedure(self):
        reached_termination = False
        while not reached_termination:
            self._record_next_observations()
            reached_termination = self._termination_criteria_satisfied()

    def _record_next_observations(self):
        new_observations = np.array([population.generate_outcome() for population in self.populations])
        self.counts = self.counts + new_observations

    def _termination_criteria_satisfied(self):
        sorted_counts = np.sort(self.counts)
        inverse_indifference = 1 / self.indifference
        confidence_value = (1 - self.desired_probability) / self.desired_probability

        z_statistic = sum(pow(inverse_indifference, sorted_counts[-1] - sorted_counts[i])
                          for i in range(len(sorted_counts) - 1))

        if z_statistic <= confidence_value:
            return True
        else:
            return False
