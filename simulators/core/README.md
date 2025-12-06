# Simulator Core

シミュレータ共通のユーティリティと基底クラスを提供するパッケージです。

## 機能

- **BaseSimulator**: 全てのシミュレータの基底クラス。共通のインターフェースとヘルパーメソッドを提供します。
- **Integration**: 数値積分（Euler法、RK4法）のユーティリティ。

## 使用方法

### 車両パラメータの読み込み

```python
from core.data import VehicleParameters
from pathlib import Path

# YAMLファイルから読み込み
params = VehicleParameters.from_yaml(Path("path/to/vehicle_config.yaml"))

# パラメータへのアクセス
print(params.wheelbase)
print(params.max_velocity)
```

### シーンの読み込み

```python
from core.data import Scene
from pathlib import Path

# YAMLファイルから読み込み
scene = Scene.from_yaml(Path("path/to/scene_config.yaml"))

# 障害物リストへのアクセス
for obstacle in scene.obstacles:
    print(obstacle.id, obstacle.x, obstacle.y)
```

### シミュレータの実装

```python
from simulator_core.base import BaseSimulator
from core.data import VehicleParameters, Scene

class MySimulator(BaseSimulator):
    def __init__(
        self,
        vehicle_params: VehicleParameters | None = None,
        scene: Scene | None = None,
        initial_state=None,
        dt=0.1
    ):
        super().__init__(
            vehicle_params=vehicle_params,
            scene=scene,
            initial_state=initial_state,
            dt=dt
        )
        # ...
```
