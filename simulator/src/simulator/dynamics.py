"""Vehicle dynamics functions."""

import math
from collections import deque
from typing import TYPE_CHECKING

from core.utils.geometry import normalize_angle

from simulator.state import SimulationVehicleState

if TYPE_CHECKING:
    from core.data import VehicleParameters, VehicleState
    from shapely.geometry import Polygon


def apply_steering_response_model(
    state: SimulationVehicleState,
    command_steering: float,
    dt: float,
    params: "VehicleParameters",
    delay_buffer: deque[float],
) -> tuple[float, deque[float]]:
    """ステアリング応答モデルを適用.

    計算順序:
    1. Clamp: 入力を最大ステア角で制限
    2. Gain: ステアリングゲインを適用
    3. Delay: 時間遅延
    4. SOPDT: 2次遅れ+無駄時間モデル
    5. RateLimit: 変化率制限

    Args:
        state: 現在の状態
        command_steering: コマンドされたステアリング角 [rad]
        dt: タイムステップ [秒]
        params: 車両パラメータ
        delay_buffer: 遅延バッファ

    Returns:
        (実際のステアリング角, 更新された遅延バッファ)
    """
    # ステップ1: Clamp - 最大ステア角で制限
    max_steering = params.max_steering_angle
    clamped_steering = max(-max_steering, min(max_steering, command_steering))

    # ステップ2: Gain - ステアリングゲインを適用
    gain_steering = clamped_steering * params.steer_gain

    # ステップ3: Delay - 時間遅延
    # 遅延バッファに新しい値を追加
    delay_buffer.append(gain_steering)

    # 遅延ステップ数を計算
    delay_steps = max(1, int(params.steer_delay_time / dt))

    # バッファサイズを調整
    while len(delay_buffer) > delay_steps:
        delay_buffer.popleft()

    # 遅延された値を取得
    delayed_steering = delay_buffer[0] if len(delay_buffer) > 0 else gain_steering
    actual_steering = state.actual_steering
    y_next = actual_steering
    v_next = state.steer_rate_internal

    if hasattr(params, "steer_tau") and params.steer_tau > 0.0:
        # ステップ4: FOPDT (First Order Plus Dead Time)
        # 1次系: tau * dy/dt + y = u
        # 離散化: y_{k+1} = alpha * y_k + (1 - alpha) * u
        # alpha = exp(-dt / tau)
        tau = params.steer_tau
        alpha = math.exp(-dt / tau)
        u_delayed = delayed_steering
        
        y_next = alpha * state.actual_steering + (1 - alpha) * u_delayed
        v_next = (y_next - state.actual_steering) / dt
    else:
        # ステップ4: SOPDT (Second Order Plus Dead Time)
        # 2次系: d²y/dt² + 2ζωn dy/dt + ωn² y = ωn² u
        # 状態変数モデル:
        #   dy/dt = v
        #   dv/dt = ωn²(u - y) - 2ζωn v

        omega_n = params.steer_omega_n
        zeta = params.steer_zeta

        # Delayed input (from step 3)
        u_delayed = delayed_steering

        # Current state
        y_k = state.actual_steering
        v_k = state.steer_rate_internal

        # Semi-implicit Euler integration
        # v_{k+1} = v_k + (omega_n^2 * (u - y_k) - 2*zeta*omega_n * v_k) * dt
        # y_{k+1} = y_k + v_{k+1} * dt

        dv_dt = (omega_n**2) * (u_delayed - y_k) - (2 * zeta * omega_n) * v_k
        v_next = v_k + dv_dt * dt
        y_next = y_k + v_next * dt

    # ステップ5: RateLimit - 変化率制限 (出力のレート制限)

    max_steer_rate_rad = params.max_steer_rate
    max_delta = max_steer_rate_rad * dt

    delta_steering = y_next - state.actual_steering

    if abs(delta_steering) > max_delta:
        delta_steering = math.copysign(max_delta, delta_steering)
        # レート制限にかかった場合、内部速度もリセットすべきだが、
        # 単純化のため位置のみ制限し、速度は次ステップで補正されるに任せる(あるいは整合させる)
        # ここでは積分整合性を保つため、速度を制限された値に合わせる
        v_next = delta_steering / dt

    actual_steering = state.actual_steering + delta_steering

    return actual_steering, v_next, delay_buffer


def update_bicycle_model(
    state: SimulationVehicleState,
    steering: float,
    acceleration: float,
    dt: float,
    params: "VehicleParameters",
) -> SimulationVehicleState:
    """Update state using kinematic bicycle model.

    Args:
        state: Current simulation state
        steering: Steering angle [rad]
        acceleration: Acceleration [m/s^2]
        dt: Time step [s]
        wheelbase: Vehicle wheelbase [m]

    Returns:
        Updated state
    """
    # Kinematic bicycle model equations
    # x_dot = vx * cos(yaw) - vy * sin(yaw)
    # y_dot = vx * sin(yaw) + vy * cos(yaw)
    # yaw_dot = vx / L * tan(delta)
    # vx_dot = ax
    
    # Apply Longitudinal Dynamics
    # Effective Acceleration = K*cmd + Offset - Drag*v^2 - CornerDrag*|steer|*v^2
    v = state.vx
    acc_eff = params.accel_gain * acceleration + params.accel_offset \
              - params.drag_coefficient * v * abs(v) \
              - params.cornering_drag_coefficient * abs(steering) * (v**2)

    # Kinamatics model assumes vy = 0
    vx_next = state.vx + acc_eff * dt

    # Use average velocity for position update
    vx_avg = (state.vx + vx_next) / 2.0

    # Calculate yaw rate (using average velocity)
    if abs(vx_avg) < 0.01:
        yaw_rate_next = 0.0
    else:
        yaw_rate_next = vx_avg / params.wheelbase * math.tan(steering)

    # Update position and orientation (using average velocity)
    x_next = state.x + vx_avg * math.cos(state.yaw) * dt
    y_next = state.y + vx_avg * math.sin(state.yaw) * dt
    yaw_next = normalize_angle(state.yaw + yaw_rate_next * dt)

    # Update timestamp if present
    timestamp_next: float | None = None
    if state.timestamp is not None:
        timestamp_next = state.timestamp + dt

    return SimulationVehicleState(
        # Position (2D update, z=0 fixed)
        x=x_next,
        y=y_next,
        z=0.0,
        # Orientation (yaw only, roll=pitch=0 fixed)
        roll=0.0,
        pitch=0.0,
        yaw=yaw_next,
        # Velocity (vx only, vy=vz=0 fixed)
        vx=vx_next,
        vy=0.0,
        vz=0.0,
        # Angular velocity (yaw_rate only)
        roll_rate=0.0,
        pitch_rate=0.0,
        yaw_rate=yaw_rate_next,
        # Acceleration
        ax=acc_eff,
        ay=0.0,
        az=0.0,
        # Input
        steering=steering,
        throttle=0.0,
        # Preserve steering response state from previous state
        actual_steering=state.actual_steering,
        target_steering=state.target_steering,
        steer_rate_internal=state.steer_rate_internal,
        # Timestamp
        timestamp=timestamp_next,
    )


def create_vehicle_polygon(
    x: float,
    y: float,
    yaw: float,
    front_edge_dist: float,
    rear_edge_dist: float,
    half_width: float,
) -> "Polygon":
    """Create vehicle polygon from parameters.

    Args:
        x: Reference point X
        y: Reference point Y
        yaw: Yaw angle
        front_edge_dist: Distance to front edge from reference
        rear_edge_dist: Distance to rear edge from reference
        half_width: Half width of vehicle

    Returns:
        Shapely Polygon
    """
    from shapely.geometry import Polygon

    # Vehicle frame coordinates (x forward, y left)
    p1 = (front_edge_dist, half_width)
    p2 = (front_edge_dist, -half_width)
    p3 = (rear_edge_dist, -half_width)
    p4 = (rear_edge_dist, half_width)

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    points = []
    for px, py in [p1, p2, p3, p4]:
        # Rotate
        rx = px * cos_yaw - py * sin_yaw
        ry = px * sin_yaw + py * cos_yaw
        # Translate
        tx = rx + x
        ty = ry + y
        points.append((tx, ty))

    return Polygon(points)


def get_bicycle_model_polygon(
    state: "VehicleState",
    params: "VehicleParameters",
) -> "Polygon":
    """Get vehicle polygon for kinematic model (rear-axle centered)."""
    # Kinematic model is rear-axle centered
    front_edge = params.wheelbase + params.front_overhang
    rear_edge = -params.rear_overhang

    return create_vehicle_polygon(
        x=state.x,
        y=state.y,
        yaw=state.yaw,
        front_edge_dist=front_edge,
        rear_edge_dist=rear_edge,
        half_width=params.width / 2.0,
    )
