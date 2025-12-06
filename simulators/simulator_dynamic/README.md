# Dynamic Simulator

車両動力学（Vehicle Dynamics）を考慮した2D車両シミュレータです。
タイヤの非線形特性や慣性を考慮し、高速域や急旋回時の挙動（横滑りなど）を再現します。

## 機能

- **DynamicVehicleModel**: 運動方程式に基づく車両モデル。
    - 4次のルンゲ=クッタ法（RK4）による高精度な数値積分。
    - 3次元状態管理（`SimulationVehicleState`）に対応。
    - 横滑り角、Yawレートのダイナミクスを計算。

## テスト仕様

`tests/test_dynamic.py` において、以下の物理挙動を検証しています。

### 1. 基本的な物理挙動
- **直進と抗力 (`test_straight_line_low_speed`)**:
    - 直進時の速度ベクトルの整合性。
    - 空気抵抗や転がり抵抗による自然減速（コースティング時の減速）の確認。
- **加速 (`test_acceleration`)**: スロットル入力に対する正の加速度の発生。

### 2. ダイナミクスの検証
- **横滑りの発生 (`test_lateral_slip`)**:
    - 高速域で急操舵を行った際に、車体の横滑り速度（`vy`）が発生することを確認。キネマティックモデルとの最大の違いです。
- **ブレーキ挙動 (`test_braking`)**:
    - 負のスロットル入力（ブレーキ）に対して、自然減速（コースティング）よりも強い減速が発生することを確認。

### 3. シミュレーター統合テスト (`TestDynamicSimulator`)
- **RK4積分**: 時間刻み `dt` での積分の安定性と動作確認。
- **初期化・ステップ実行**: シミュレータとしての基本的なインターフェース動作。

## 使用方法

```python
from simulator_dynamic import DynamicSimulator
from core.data import Action

# RK4を使用するため、比較的小さなdtを推奨
sim = DynamicSimulator(dt=0.01)
sim.reset()

action = Action(steering=0.1, acceleration=1.0)
next_state, done, info = sim.step(action)
```
