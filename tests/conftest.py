"""Shared fixtures for yippy tests."""

import pytest

from yippy import Coronagraph


@pytest.fixture(scope="session")
def coro():
    """Session-scoped real coronagraph loaded from coronalyze's pooch data."""
    from coronalyze.datasets import fetch_coronagraph

    yip_path = fetch_coronagraph()
    return Coronagraph(yip_path)


@pytest.fixture(scope="session")
def eqx_coro(coro):
    """Session-scoped EqxCoronagraph built from the real coronagraph."""
    from yippy.eqx_coronagraph import EqxCoronagraph

    return EqxCoronagraph(yippy_coro=coro)
