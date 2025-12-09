#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Run all tests with coverage
PYTHONPATH=. pytest --cov=app --cov-report=term-missing tests/
