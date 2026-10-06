"""Shared pytest fixtures: the sample repo and its hand-checked expected outputs."""

import json
from pathlib import Path

import pytest

from tokencut.contracts import Chunk, CodeGraph

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="session")
def sample_repo_path() -> Path:
    return FIXTURES / "sample_repo"


@pytest.fixture(scope="session")
def expected_chunks() -> list[Chunk]:
    with open(FIXTURES / "expected_chunks.json", encoding="utf-8") as f:
        return [Chunk.model_validate(x) for x in json.load(f)]


@pytest.fixture(scope="session")
def expected_graph() -> CodeGraph:
    with open(FIXTURES / "expected_graph.json", encoding="utf-8") as f:
        return CodeGraph.model_validate(json.load(f))
