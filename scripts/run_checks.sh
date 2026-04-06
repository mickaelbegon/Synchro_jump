#!/usr/bin/env bash
set -euo pipefail

isort . --check-only --profile black
black . --check
flake8 .
pytest -q
