"""Tests for solver module."""

from simulator_core.solver import rk4_step


class TestSolver:
    """Tests for numerical solvers."""

    def test_rk4_step_linear(self) -> None:
        """Test RK4 with a simple linear function: dx/dt = 1."""

        # dx/dt = 1, x(0) = 0 -> x(t) = t
        def derivative_func(state: float) -> float:
            return 1.0

        def add_func(state: float, derivative: float, dt: float) -> float:
            return state + derivative * dt

        state = 0.0
        dt = 0.1

        # Step 1: t=0.1
        state = rk4_step(state, derivative_func, dt, add_func)
        assert abs(state - 0.1) < 1e-10

        # Step 2: t=0.2
        state = rk4_step(state, derivative_func, dt, add_func)
        assert abs(state - 0.2) < 1e-10

    def test_rk4_step_exponential(self) -> None:
        """Test RK4 with exponential function: dx/dt = x."""

        # dx/dt = x, x(0) = 1 -> x(t) = e^t
        def derivative_func(state: float) -> float:
            return state

        def add_func(state: float, derivative: float, dt: float) -> float:
            return state + derivative * dt

        state = 1.0
        dt = 0.1

        # Exact value at t=0.1 is e^0.1 approx 1.1051709
        expected = 1.105170918

        state = rk4_step(state, derivative_func, dt, add_func)

        # RK4 error is O(dt^5), so it should be very precise
        assert abs(state - expected) < 1e-6

    def test_rk4_step_vector(self) -> None:
        """Test RK4 with vector state (list)."""
        # dx/dt = y, dy/dt = -x (Simple Harmonic Oscillator)
        # x(0) = 0, y(0) = 1 -> x(t) = sin(t), y(t) = cos(t)

        def derivative_func(state: list[float]) -> list[float]:
            x, y = state
            return [y, -x]

        def add_func(state: list[float], derivative: list[float], dt: float) -> list[float]:
            return [state[0] + derivative[0] * dt, state[1] + derivative[1] * dt]

        state = [0.0, 1.0]
        dt = 0.1

        state = rk4_step(state, derivative_func, dt, add_func)

        # Analytical solution at t=0.1
        import math

        expected_x = math.sin(0.1)
        expected_y = math.cos(0.1)

        assert abs(state[0] - expected_x) < 1e-6
        assert abs(state[1] - expected_y) < 1e-6
