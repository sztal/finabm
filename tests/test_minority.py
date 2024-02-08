"""Test Minority Game."""
# pylint: disable=missing-function-docstring,redefined-outer-name
import numpy as np
import pytest
from finabm import MinorityGame


@pytest.fixture(scope="function")
def mg_simple():
    # pylint: disable=protected-access
    N = 3
    S = 2
    M = np.array([3, 2, 2])
    T = 5
    mg = MinorityGame(N, M, S)
    mg._attendance[:] = [1, -1, 1]
    mg.V[:, 0] = 1
    mg.P[0, :] = [-1, -1, 1, 1, 1, -1, -1, 1]
    mg.P[1, :] = [1, -1, 1, 1, -1, -1, 1, 1]
    mg.P[2, :4] = [-1, 1, -1, 1]
    mg.P[3, :4] = [1, 1, -1, -1]
    mg.P[4, :4] = [1, 1, -1, 1]
    mg.P[5, :4] = [-1, 1, 1, 1]
    mg.mg._init_C()
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
        [ 1,  1, -1, -1, -1],
        [ 1,  1,  1,  1,  1]
    ])
    # Strategy scores
    V = np.array([
        [ 2,  1,  0, -1, -2],
        [ 1,  0, -1, -2, -3],
        [ 0, -1, -2, -3, -4],
        [-1,  0,  1,  2,  3],
        [ 0, -1, -2, -3, -4],
        [-1, -2, -3, -4, -5]
    ])
    # Agent scores
    W = np.array([
        [ 1,  0, -1, -2, -3],
        [-1, -2, -1,  0,  1],
        [-1, -2, -3, -4, -5]
    ])
    attendance = np.array([1, 3, 1, 1, 1])
    history = np.array([1, -1, 1, 1, 1, 1, 1, 1])
    return mg, T, A_strategies, A_agents, V, W, attendance, history


def test_mg_simple(mg_simple):
    """Test MG for a simple scenario."""
    mg, T, A_strategies, A_agents, V, W, attendance, history = mg_simple
    mg.prepare(T)
    for i in range(T):
        _, agent_decisions, strategy_decisions = mg.step()
        assert np.array_equal(A_agents[:, i], agent_decisions)
        assert np.array_equal(A_strategies[:, i], strategy_decisions)
        assert np.array_equal(V[:, i], mg.V.flatten())
        assert np.array_equal(W[:, i], mg.scores)
    assert np.array_equal(attendance, mg.attendance)
    assert np.array_equal(history, mg.full_history)
