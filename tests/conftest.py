"""Shared pytest configuration for AI Document Enhancement System."""

import os
import sys

# Ensure the application package is importable from the project root
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
