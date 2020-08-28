# simulation-methods
Package containing some simple models and methods used for simulation


## General Use

The library currently has support for bernoulli ranking and selection procedures, system output comparison, and simple
simulation models.

You can install the package by running the following command: \
`pip install git+https://github.com/bsatterwhite3/simulation-methods.git`

### Example Use
The following example shows how to run one of the Bernoulli selection procedures. These
procedures require the caller to create an interface to their system by extending the 
BernoulliPopulation and implementing a generate_outcome method that returns a 1 or a 0. This example will take 1000 samples
max from each population to try to determine a winner. 

```
from simulation_methods.procedures import GenericBernoulliPopulation, BernoulliBechhoferKulkarni

population_A = GenericBernoulliPopulation(prob=0.2)
population_B = GenericBernoulliPopulation(prob=0.8)

bernoulli_bk = BernoulliBechhoferKulkarni([population_A, population_B], max_samples=1000)
winner = bernoulli_bk.select_best_population()  # Winner would most likely be population_B because 80% >> 20%
```

## Development

### Project Structure

- `procedures.py`: contains implementations from various papers on ranking/selection procedures
- `methods.py`: contains methods for comparing outputs of simulations
- `models.py`: contains models that can be used for simulation, like Brownian Motion

### Installing

To install the application for local development:
1. Create virtual environment of your choice with Python 3.7 installed (`conda`, `virtualenv`, `pyenv`, etc.)
2. Run `python setup.py install`

This will install the project on your local machine along with the libraries listed in the `requirements.txt` file.

### Testing

Once the project is installed locally, run `pip install -r test-requirements.txt` to install the packages for testing.

Then run `python -m pytest tests` to run the tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details

## Papers Referenced

* [On the Performance Characteristics of a Closed Adaptive Sequential Procedure for Selecting the Best Bernoulli Population.](https://apps.dtic.mil/sti/citations/ADA115219)
* [Sequential Identification and Ranking Procedures: With Special Reference to Koopman-Darmois Populations](https://books.google.com/books/about/Sequential_identification_and_ranking_pr.html?id=ixC3rvnHK5sC)

