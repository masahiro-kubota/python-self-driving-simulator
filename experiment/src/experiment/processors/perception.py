from core.data import Observation, VehicleState


class BasicPerceptionProcessor:
    """基本的な認識プロセッサー."""

    def process(self, vehicle_state: VehicleState) -> Observation:
        """車両状態から観測情報を生成 (現在はダミー実装)."""
        return Observation(
            lateral_error=0.0,
            heading_error=0.0,
            velocity=vehicle_state.velocity,
            target_velocity=0.0,
            timestamp=vehicle_state.timestamp,
        )
