from core.data import VehicleState


class IdealSensorProcessor:
    """理想的なセンサープロセッサー (ノイズなし、遅延なし)."""

    def process(self, sim_state: VehicleState) -> VehicleState:
        """シミュレーション状態をそのまま車両状態として出力."""
        return sim_state
