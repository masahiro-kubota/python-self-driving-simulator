from core.data import VehicleParameters
from pure_pursuit.pure_pursuit_node import PurePursuitConfig, PurePursuitNode


def test_pure_pursuit_node_instantiation(tmp_path) -> None:
    """Configオブジェクトを直接生成してPurePursuitNodeが正しくインスタンス化できるかテスト"""

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

    # ダミーのトラックファイル作成
    dummy_track = tmp_path / "test_track.csv"
    dummy_track.write_text(
        "x,y,z,x_quat,y_quat,z_quat,w_quat,speed\n0,0,0,0,0,0,1,10\n10,0,0,0,0,0,1,10"
    )

    # Configの直接生成
    config = PurePursuitConfig(
        track_path=str(dummy_track),
        min_lookahead_distance=2.0,
        max_lookahead_distance=10.0,
        lookahead_speed_ratio=1.0,
        vehicle_params=vp,
    )

    # ノードのインスタンス化
    node = PurePursuitNode(config=config, rate_hz=10.0, priority=10)

    assert node.name == "PurePursuit"
    assert str(node.config.track_path) == str(dummy_track)
    assert node.config.min_lookahead_distance == 2.0
