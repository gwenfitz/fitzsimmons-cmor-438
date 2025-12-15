# Tests
This directory contains unit tests for the rice_ml machine learning package. All tests are written using pytest and validate core functionality, correctness, and basic edge cases for each algorithm.

## Organization

Test files are named to correspond directly to their implementation modules (e.g., test_pca.py, test_logistic_regression.py).

## Design Notes

Tests use small synthetic datasets for clarity and speed.
Randomness is controlled where applicable.
Tests focus on expected behavior rather than performance benchmarking.

## Running Tests

From the project root:
`pytest`