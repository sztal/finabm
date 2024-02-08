"""Test Dollar Game."""
# pylint: disable=missing-function-docstring,redefined-outer-name
import numpy as np
import pytest
from finabm import DollarGame


@pytest.fixture(scope="function")
def dg_simple():
    # pylint: disable=protected-access
    N = 3
    S = 2
    M = np.array([3, 2, 2])
    T = 5
    dg = DollarGame(N, M, S, liquidity=1)
    dg._attendance[:] = [1, -1, 1]
    dg._price = np.array([ dg.initial_price, *dg.get_price(dg._attendance) ])
    dg.dg.V = np.array([[1, 2], [0, 1], [2, 0]], dtype=dg.V.dtype)
    dg.P[0, :] = [-1, -1, 1, 1, 1, -1, -1, 1]
    dg.P[1, :] = [1, -1, 1, 1, -1, -1, 1, 1]
    dg.P[2, :4] = [-1, 1, -1, 1]
    dg.P[3, :4] = [1, 1, -1, -1]
    dg.P[4, :4] = [1, 1, -1, 1]
    dg.P[5, :4] = [-1, 1, 1, 1]
    dg.dg._init_C()
    # All strategy actions
    # | subsequent time steps in columns
    A_strategies = np.array([
        [-1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1]
    ])
    # Actions selected by agents
    A_agents = np.array([
        [-1,  1,  1,  1,  1],
        [ 1, -1, -1,  1,  1],
        [ 1,  1,  1,  1,  1]
    ])
    # Strategy scores
    V = np.array([
        [ 1,  1,  0,  1,  4],
        [ 2,  2,  1,  2,  5],
        [ 0,  0,  1,  2,  5],
        [ 1,  1,  2,  1, -2],
        [ 2,  2,  3,  4,  7],
        [ 0,  0,  1,  2,  5]
    ])
    # Agent scores
    W = np.array([
        [ 0,  0, -1,  0,  3],
        [ 0,  0,  1,  0, -3],
        [ 0,  0,  1,  2,  5]
    ])
    attendance = np.array([1, 1, 1, 3, 3])
    history = np.array([1, -1, 1, 1, 1, 1, 1, 1])
    return dg, T, A_strategies, A_agents, V, W, attendance, history


def test_dg_simple(dg_simple):
    """Test $G for a simple scenario."""
    dg, T, A_strategies, A_agents, V, W, attendance, history = dg_simple
    dg.prepare(T)
    for i in range(T):
        _, agent_decisions, strategy_decisions = dg.step()
        assert np.array_equal(A_agents[:, i], agent_decisions)
        assert np.array_equal(A_strategies[:, i], strategy_decisions)
        assert np.array_equal(V[:, i], dg.V.flatten())
        assert np.array_equal(W[:, i], dg.scores)
    assert np.array_equal(attendance, dg.attendance)
    assert np.array_equal(history, dg.full_history)
