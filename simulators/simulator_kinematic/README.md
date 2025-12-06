# Kinematic Simulator

自転車モデル（Kinematic Bicycle Model）に基づく軽量な2D車両シミュレータです。
低速域およびタイヤのグリップ限界内での車両挙動を近似するのに適しています。

## 機能

- **KinematicVehicleModel**: 幾何学的な拘束条件に基づく車両モデル。
    - 入力: ステアリング角、加速度
    - 状態: 位置(x, y)、方位角(yaw)、速度(v)
    - 後輪中心を基準点としています。

## テスト仕様

`tests/test_kinematic.py` において、以下の動作検証を行っています。

### 1. 基本動作の検証
- **直進 (`test_straight_line`)**: ステアリング0で直進し、横方向の変位がないこと。
- **加速 (`test_acceleration`)**: 指定した加速度で速度が増加すること。
- **旋回 (`test_turning`)**: ステアリングを切った状態でYaw角が変化し、理論値と一致すること。

### 2. 特殊な挙動の検証
- **後退 (`test_backward_motion`)**:
    - 負の速度において位置が減少すること。
    - ステアリング操作に対する旋回方向（Yawレートの符号）が正しいこと（後退時も前進時と同じ幾何学的拘束に従う）。

### 3. シミュレーター統合テスト (`TestKinematicSimulator`)
- **初期化**: デフォルトパラメータおよびカスタム初期状態で正しく初期化されるか。
- **ステップ実行**: `Action` を受け取り、状態を更新して返す一連の流れ。

## 使用方法

```python
from simulator_kinematic import KinematicSimulator
from core.data import Action

sim = KinematicSimulator(dt=0.1)
sim.reset()

action = Action(steering=0.1, acceleration=1.0)
next_state, done, info = sim.step(action)
```
