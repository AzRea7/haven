# tests/conftest.py
import pytest
from fastapi.testclient import TestClient

from haven.api.http import app  # ensures imports resolve; run tests from repo root


@pytest.fixture(scope="session")
def client():
    return TestClient(app)
