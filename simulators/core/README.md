# Simulator Core

シミュレータ共通のユーティリティと基底クラスを提供するパッケージです。

## 機能

- **BaseSimulator** (`simulator.py`): 全てのシミュレータの基底クラス。共通のインターフェース、ロギング、ステップ実行フロー、ゴール判定を提供します。
- **Solver** (`solver.py`): 数値積分（RK4法）のユーティリティ。
- **DynamicVehicleState** (`data/state.py`): シミュレータ内部で使用される、完全な3D車両状態データ構造。
- **LaneletMap** (`map/lanelet_map.py`): Lanelet2マップの読み込みと判定（走行可能領域、境界チェックなど）を行うユーティリティ。

## テスト仕様

本パッケージのテストは、シミュレーション基盤の信頼性を保証するために以下の観点で実施されています。

### 1. 基底クラスのテスト (`tests/test_simulator.py`)
`BaseSimulator` のライフサイクルと共通機能を検証します。
- **初期化**: パラメータ、初期状態が正しく設定されるか。
- **ステップ実行**: `step` メソッドがサブクラスの `_update_state` を呼び出し、ログを記録し、状態を変換して返すフロー。
- **実行ループ**: `run` メソッドが指定ステップ数またはゴール到達までループを実行するか。
- **ゴール判定**: 参照軌道の終点との距離および経過時間に基づくゴール判定ロジック。

### 2. 数値積分のテスト (`tests/test_solver.py`)
4次のルンゲ=クッタ法 (`rk4_step`) の精度を検証します。
- **線形関数**: `dx/dt = 1` (厳密解と一致することを確認)
- **指数関数**: `dx/dt = x` (指数挙動の精度確認)
- **ベクトル状態**: 状態変数がリスト/ベクトルである場合の動作確認（単振動など）。

### 3. 状態変換のテスト (`tests/test_state.py`)
`DynamicVehicleState` と外部向け `VehicleState` の相互変換を検証します。
- **from_vehicle_state**: 2D状態から3D状態への拡張（不足パラメータのゼロ埋め、速度ベクトルの分解など）。
- **to_vehicle_state**: 3D状態から2D状態への縮退（アクション適用後のステアリング/加速度の上書き確認）。

### 4. マップ機能のテスト (`tests/test_lanelet_map.py`, `tests/test_map_integration.py`)
- **マップ読み込み**: Lanelet2ファイルのパース。
- **走行可能領域判定**: 指定座標がレーン内にあるかどうか (`is_drivable`)。
- **統合テスト**: シミュレーターにマップをロードした状態で、コースアウト判定が機能するか。

## 使用方法

### シミュレータの実装

```python
from simulator_core.simulator import BaseSimulator
from simulator_core.data import DynamicVehicleState
from core.data import Action, VehicleParameters, VehicleState

class MySimulator(BaseSimulator):
    def __init__(self, vehicle_params=None, initial_state=None, dt=0.1, map_path=None):
        super().__init__(vehicle_params, initial_state, dt, map_path)

    def _update_state(self, action: Action) -> DynamicVehicleState:
        # ここで状態更新ロジックを実装
        # self._current_state (DynamicVehicleState) を使用
        # 新しい DynamicVehicleState を返す
        pass
```
