"""Linear MPC solver for lateral control using kinematic bicycle model."""

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MPCConfig:
    """Configuration for MPC solver."""

    prediction_horizon: int  # Prediction horizon [steps]
    control_horizon: int  # Control horizon [steps]
    dt: float  # Discretization time step [s]

    # Weights
    weight_lateral_error: float  # Weight for lateral error
    weight_heading_error: float  # Weight for heading error
    weight_steering: float  # Weight for steering input
    weight_steering_rate: float  # Weight for steering rate

    # Constraints
    max_steering_angle: float  # Maximum steering angle [rad]
    max_steering_rate: float  # Maximum steering rate [rad/s]


class LinearMPCLateralSolver:
    """Linear MPC solver for lateral path tracking.

    Uses a simplified kinematic bicycle model linearized around the reference path.
    State: [lateral_error, heading_error]
    Input: [steering_angle]
    """

    def __init__(self, config: MPCConfig, wheelbase: float):
        """Initialize MPC solver.

        Args:
            config: MPC configuration
            wheelbase: Vehicle wheelbase [m]
        """
        self.config = config
        self.wheelbase = wheelbase

        # Store previous solution for warm start
        self.previous_steering = 0.0

    def solve(
        self,
        lateral_error: float,
        heading_error: float,
        current_steering: float,
        reference_curvature: np.ndarray,
        current_velocity: float,
    ) -> tuple[float, bool]:
        """Solve MPC optimization problem.

        Args:
            lateral_error: Current lateral error [m]
            heading_error: Current heading error [rad]
            current_steering: Current steering angle [rad]
            reference_curvature: Reference path curvature for prediction horizon [1/m]
            current_velocity: Current vehicle velocity [m/s]

        Returns:
            tuple: (optimal_steering_angle, predicted_states, predicted_controls, success)
                - optimal_steering_angle: Optimal steering angle [rad]
                - predicted_states: Predicted state trajectory [2, N+1]
                - predicted_controls: Predicted control trajectory [1, M]
                - success: Whether optimization succeeded
        """
        N = self.config.prediction_horizon
        M = self.config.control_horizon
        dt = self.config.dt

        # Ensure velocity is not too small to avoid numerical issues
        v = max(abs(current_velocity), 0.1)

        # Ensure reference curvature has correct length
        if len(reference_curvature) < N:
            # Extend with last value
            kappa = np.concatenate(
                [reference_curvature, np.full(N - len(reference_curvature), reference_curvature[-1])]
            )
        else:
            kappa = reference_curvature[:N]

        # Build linearized state-space model around reference
        # State: x = [e_y, e_psi]
        # Input: u = [delta]
        #
        # Simplified lateral dynamics (small angle approximation):
        # e_y_dot = v * sin(e_psi) ≈ v * e_psi
        # e_psi_dot = v * tan(delta) / L - v * kappa ≈ v * delta / L - v * kappa

        # Decision variables
        x = cp.Variable((2, N + 1))  # State trajectory: [e_y, e_psi]
        u = cp.Variable((1, M))  # Control input: [delta]

        # Cost function
        cost = 0.0

        # State cost (tracking error)
        for k in range(N):
            # Lateral error cost
            cost += self.config.weight_lateral_error * cp.square(x[0, k])
            # Heading error cost
            cost += self.config.weight_heading_error * cp.square(x[1, k])

        # Control input cost
        for k in range(M):
            cost += self.config.weight_steering * cp.square(u[0, k])

        # Control rate cost (smoothness)
        for k in range(M - 1):
            cost += self.config.weight_steering_rate * cp.square(u[0, k + 1] - u[0, k])

        # Constraints
        constraints = []

        # Initial state constraint
        constraints.append(x[0, 0] == lateral_error)
        constraints.append(x[1, 0] == heading_error)

        # Dynamics constraints (using forward Euler discretization)
        for k in range(N):
            # Determine control input for this timestep
            if k < M:
                u_k = u[0, k]
            else:
                # After control horizon, keep last control
                u_k = u[0, M - 1]

            # Linearized dynamics
            # e_y(k+1) = e_y(k) + dt * v * e_psi(k)
            constraints.append(x[0, k + 1] == x[0, k] + dt * v * x[1, k])

            # e_psi(k+1) = e_psi(k) + dt * (v * delta(k) / L - v * kappa(k))
            constraints.append(
                x[1, k + 1] == x[1, k] + dt * (v * u_k / self.wheelbase - v * kappa[k])
            )

        # Control constraints (steering angle limits)
        constraints.append(u <= self.config.max_steering_angle)
        constraints.append(u >= -self.config.max_steering_angle)

        # Solve optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                optimal_steering_angle = u[0, 0].value
                predicted_states = x.value
                predicted_controls = u.value
                
                logger.debug(f"[MPC Solver] Status: {problem.status}, Cost: {problem.value:.4f}")
                logger.debug(f"[MPC Solver] Optimal steering angle: {optimal_steering_angle:.6f} rad")
                logger.debug(f"[MPC Solver] Predicted states x[:,0]: lat_err={x[0,0].value:.3f}, "
                            f"head_err={x[1,0].value:.3f}")
                
                return float(optimal_steering_angle), predicted_states, predicted_controls, True
            else:
                logger.warning(f"[MPC Solver] Optimization failed with status: {problem.status}")
                return current_steering, None, None, False

        except Exception as e:
            logger.error(f"[MPC Solver] Optimization error: {e}")
            return current_steering, None, None, False
