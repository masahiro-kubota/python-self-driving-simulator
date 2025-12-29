from core.data import VehicleParameters
from pid_controller.pid_controller_node import PIDConfig, PIDControllerNode


def test_pid_controller_node_instantiation() -> None:
    """Configオブジェクトを直接生成してPIDControllerNodeが正しくインスタンス化できるかテスト"""

    # ダミーの車両パラメータ
    vp = VehicleParameters(
        wheelbase=2.5,
        width=1.8,
        max_steering_angle=0.6,
        max_velocity=20.0,
        max_acceleration=3.0,
        mass=1500.0,
        inertia=2500.0,
        lf=1.2,
        lr=1.3,
        cf=80000.0,
        cr=80000.0,
        c_drag=0.3,
        c_roll=0.015,
        max_drive_force=5000.0,
        max_brake_force=8000.0,
        front_overhang=1.0,
        rear_overhang=1.0,
        vehicle_height=1.5,
        tire_params={"type": "test_tire"},
    )

    # Configの直接生成
    config = PIDConfig(kp=1.0, ki=0.1, kd=0.01, u_min=-10.0, u_max=10.0, vehicle_params=vp)

    # ノードのインスタンス化
    node = PIDControllerNode(config=config, rate_hz=30.0, priority=10)

    assert node.name == "PIDController"
    assert node.config.kp == 1.0
