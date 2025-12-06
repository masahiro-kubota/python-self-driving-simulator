"""Numerical integration methods for simulators."""

from collections.abc import Callable
from typing import TypeVar

StateT = TypeVar("StateT")
DerivativeFunc = Callable[[StateT], StateT]


def rk4_step(
    state: StateT,
    derivative_func: DerivativeFunc[StateT],
    dt: float,
    add_func: Callable[[StateT, StateT, float], StateT],
) -> StateT:
    """Runge-Kutta 4次法による1ステップ積分.

    Args:
        state: 現在の状態
        derivative_func: 状態の微分を計算する関数
        dt: 時間刻み [s]
        add_func: 状態に微分を加算する関数 (state, derivative, dt) -> new_state

    Returns:
        更新された状態
    """
    k1 = derivative_func(state)
    state2 = add_func(state, k1, dt / 2)

    k2 = derivative_func(state2)
    state3 = add_func(state, k2, dt / 2)

    k3 = derivative_func(state3)
    state4 = add_func(state, k3, dt)

    k4 = derivative_func(state4)

    # Accumulate changes
    # y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    # We apply the updates sequentially
    new_state = add_func(state, k1, dt / 6)
    new_state = add_func(new_state, k2, dt / 3)
    new_state = add_func(new_state, k3, dt / 3)
    new_state = add_func(new_state, k4, dt / 6)

    return new_state
