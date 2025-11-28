"""
Pytest configuration and fixtures for tests.

This file is automatically loaded by pytest and provides shared fixtures
and configuration for all tests.
"""

import os
import pytest

# Set dummy API key before any imports that require it
# This prevents errors when importing modules that initialize OpenAI clients
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")


@pytest.fixture(autouse=True)
def ensure_api_key():
    """Ensure OPENAI_API_KEY is set for all tests."""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
    yield
    # Cleanup if needed (optional)
