import pytest
import numpy as np

from simulation_methods import procedures


class TestBernoulliSelection(object):
    bern_pop_A = procedures.GenericBernoulliPopulation(.9)
    bern_pop_B = procedures.GenericBernoulliPopulation(.2)
    bern_pop_C = procedures.GenericBernoulliPopulation(.5)


class TestBernoulliBechhoferKulkarni(TestBernoulliSelection):
    def test_select_best_population(self, mocker):
        mock_bernoulli = mocker.patch.object(self.bern_pop_B, 'generate_outcome')
        mock_bernoulli.return_value = 1  # Ensures that no other populations will outperform this one

        bernoulli_bk = procedures.BernoulliBechhoferKulkarni([self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 10000)
        assert bernoulli_bk.select_best_population() == 1

    def test_initialize_procedure(self):
        expected = {
            0: procedures.BernoulliBechhoferKulkarni.ObservationCounter(0, 0, 0),
            1: procedures.BernoulliBechhoferKulkarni.ObservationCounter(0, 0, 0),
        }

        bernoulli_bk = procedures.BernoulliBechhoferKulkarni([self.bern_pop_A, self.bern_pop_B], 100)
        result = bernoulli_bk._initialize_procedure()
        assert result == expected

    def test_get_index_of_lowest_failure(self):
        observation_map = {
            0: procedures.BernoulliBechhoferKulkarni.ObservationCounter(30, 25, 5),
            1: procedures.BernoulliBechhoferKulkarni.ObservationCounter(25, 7, 18),
            2: procedures.BernoulliBechhoferKulkarni.ObservationCounter(24, 12, 12),
        }

        bernoulli_bk = procedures.BernoulliBechhoferKulkarni([self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 100)
        assert bernoulli_bk._get_index_of_lowest_failure() == -1

        bernoulli_bk.observation_map = observation_map
        assert bernoulli_bk._get_index_of_lowest_failure() == 0

    def test_get_index_of_highest_success(self):
        observation_map = {
            0: procedures.BernoulliBechhoferKulkarni.ObservationCounter(30, 25, 5),
            1: procedures.BernoulliBechhoferKulkarni.ObservationCounter(25, 7, 18),
            2: procedures.BernoulliBechhoferKulkarni.ObservationCounter(24, 12, 12),
        }

        bernoulli_bk = procedures.BernoulliBechhoferKulkarni([self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 30)
        bernoulli_bk.observation_map = observation_map
        assert bernoulli_bk._get_index_of_highest_success(include_terminated_populations=False) in [1, 2]

        bernoulli_bk.observation_map = observation_map
        assert bernoulli_bk._get_index_of_highest_success(include_terminated_populations=True) == 0

    def test_get_next_population_by_failures(self):
        observation_map = {
            0: procedures.BernoulliBechhoferKulkarni.ObservationCounter(30, 25, 5),
            1: procedures.BernoulliBechhoferKulkarni.ObservationCounter(25, 7, 18),
            2: procedures.BernoulliBechhoferKulkarni.ObservationCounter(24, 12, 12),
        }

        bernoulli_bk = procedures.BernoulliBechhoferKulkarni([self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 100)
        bernoulli_bk.observation_map = observation_map

        assert bernoulli_bk._get_index_of_highest_success() == 0

    def test_get_next_population_by_successes(self):
        observation_map = {
            0: procedures.BernoulliBechhoferKulkarni.ObservationCounter(43, 25, 18),
            1: procedures.BernoulliBechhoferKulkarni.ObservationCounter(44, 26, 18),
            2: procedures.BernoulliBechhoferKulkarni.ObservationCounter(24, 12, 12),
        }

        bernoulli_bk = procedures.BernoulliBechhoferKulkarni([self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 100)
        bernoulli_bk.observation_map = observation_map

        assert bernoulli_bk._get_index_of_highest_success() == 1

    def test_update_observation_map(self):
        expected_observation_map = {
            0: procedures.BernoulliBechhoferKulkarni.ObservationCounter(1, 1, 0),
            1: procedures.BernoulliBechhoferKulkarni.ObservationCounter(1, 0, 1),
            2: procedures.BernoulliBechhoferKulkarni.ObservationCounter(0, 0, 0)
        }

        expected_observation_history = [
            (1, procedures.BernoulliBechhoferKulkarni.ObservationCounter(1, 0, 1)),
            (0, procedures.BernoulliBechhoferKulkarni.ObservationCounter(1, 1, 0))
        ]

        bernoulli_bk = procedures.BernoulliBechhoferKulkarni([self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 100)
        bernoulli_bk._update_observation_map(1, 0)
        bernoulli_bk._update_observation_map(0, 1)

        assert bernoulli_bk.observation_map == expected_observation_map
        assert bernoulli_bk.observation_history == expected_observation_history

    def test_termination_criteria_satisfied(self):
        observation_map = {
            0: procedures.BernoulliBechhoferKulkarni.ObservationCounter(90, 80, 10),
            1: procedures.BernoulliBechhoferKulkarni.ObservationCounter(50, 10, 40),
            2: procedures.BernoulliBechhoferKulkarni.ObservationCounter(40, 18, 22),
        }

        bernoulli_bk = procedures.BernoulliBechhoferKulkarni([self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 100)
        bernoulli_bk.observation_map = observation_map

        assert bernoulli_bk._termination_criteria_satisfied()

        observation_map = {
            0: procedures.BernoulliBechhoferKulkarni.ObservationCounter(90, 80, 10),
            1: procedures.BernoulliBechhoferKulkarni.ObservationCounter(50, 10, 40),
            # Population 2 could feasibly catch up to Population 0
            2: procedures.BernoulliBechhoferKulkarni.ObservationCounter(40, 22, 18)
        }

        bernoulli_bk = procedures.BernoulliBechhoferKulkarni([self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 100)
        bernoulli_bk.observation_map = observation_map

        assert not bernoulli_bk._termination_criteria_satisfied()


class TestBernoulliBechhoferKieferSobel(TestBernoulliSelection):
    def test_select_best_population(self, mocker):

        mock_bernoulli = mocker.patch.object(self.bern_pop_B, 'generate_outcome')
        mock_bernoulli.return_value = 1  # Ensures that no other populations will outperform this one

        bernoulli_bks = procedures.BernoulliBechhoferKieferSobel(
            [self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 1.5, .95
        )
        assert bernoulli_bks.select_best_population() == 1

    def test_validate_procedure_inputs(self):
        bernoulli_bks = procedures.BernoulliBechhoferKieferSobel(
            [self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 1.5, .95
        )

        invalid_indifference = .5
        valid_indifference = 1.4

        invalid_probability = 2
        valid_probability = .65

        with pytest.raises(ValueError):
            bernoulli_bks._validate_procedure_inputs(invalid_indifference, valid_probability)

        with pytest.raises(ValueError):
            bernoulli_bks._validate_procedure_inputs(valid_indifference, invalid_probability)

        bernoulli_bks._validate_procedure_inputs(valid_indifference, valid_probability)

    def test_run_procedure(self, mocker):
        mock_termination = mocker.patch('simulation_methods.procedures.BernoulliBechhoferKieferSobel._termination_criteria_satisfied')
        mock_termination.return_value = True

        mock_observation = mocker.patch('simulation_methods.procedures.BernoulliBechhoferKieferSobel._record_next_observations')

        bernoulli_bks = procedures.BernoulliBechhoferKieferSobel(
            [self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 1.4, .65
        )

        bernoulli_bks._run_procedure()

        mock_observation.assert_called_once()
        mock_termination.assert_called_once()

    def test_record_next_observations(self, mocker):
        bernoulli_bks = procedures.BernoulliBechhoferKieferSobel(
            [self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 1.4, .65
        )
        mocker.patch('simulation_methods.procedures.GenericBernoulliPopulation.generate_outcome', return_value=1)
        bernoulli_bks._record_next_observations()

        expected_counts = np.array([1., 1., 1.])
        assert np.array_equal(bernoulli_bks.counts, expected_counts)

    def test_termination_criteria_satisfied(self):
        bernoulli_bks = procedures.BernoulliBechhoferKieferSobel(
            [self.bern_pop_A, self.bern_pop_B, self.bern_pop_C], 1.4, .65
        )
        assert not bernoulli_bks._termination_criteria_satisfied()

        counts = np.array([4, 8, 3])
        bernoulli_bks.counts = counts
        assert bernoulli_bks._termination_criteria_satisfied()
