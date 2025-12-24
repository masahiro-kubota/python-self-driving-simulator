# Tiny LiDAR Net 開発計画

## 概要
Tiny LiDAR Netの開発に向けたデータセット作成、学習、評価の進め方についてまとめる。
初期段階として、ルールベースの制御（Pure Pursuit）を用いたImitation Learning（模倣学習）のアプローチを採用する。

## 1. データセット作成 (Dataset Generation)

### 方針
- **Expert**: Pure Pursuit
- **Environment**: ランダムに障害物が配置されたシナリオ
- **Validation**: Pure Pursuitが1周完走できたシナリオのみを「有効なデータ」として採用する

### 手順
1. **ランダムシナリオ生成**:
   - コース上に障害物をランダムな位置・向きで配置する仕組みを実装する。
   - 難易度の調整（障害物の数、配置の分散など）をパラメータ化する。

2. **データ収集 (Data Collection)**:
   - Pure Pursuitエージェントを走行させる。
   - センサデータ（LiDAR）、車両状態（位置、速度）、制御入力（操舵角、アクセル）を記録する。
   - 走行が「成功（完走）」したか「失敗（衝突/スタック）」したかのフラグを記録する。

3. **データフィルタリング**:
   - 成功したエピソードのみを抽出して学習用データセット（Train）および検証用データセット（Val）とする。

## 2. モデル学習 (Training)

### 方針
- **Model**: Tiny LiDAR Net（軽量な1D CNN等を想定）
- **Input**: LiDARスキャンデータ（Ranges）
- **Output**: 制御コマンド（Steering, Speed/Throttle）

### 手順
1. **データローダー実装**:
   - 収集したデータセットを読み込むPipelineを構築。
   - 必要に応じてData Augmentation（ノイズ付加、意図的なDrop-outなど）を検討。

2. **学習ループ**:
   - Loss関数: MSE（Mean Squared Error）などを想定。
   - Optimizer: AdamWなど。

## 3. 評価 (Evaluation)

### 方針
- 未知のシナリオ（Test Scenarios）における完走率や走行のスムーズさで評価する。

### 手順
1. **テストシナリオ作成**:
   - 学習には使用していない固定のテストシナリオセットを用意する。
   - 難易度別にレベル分け（Easy, Medium, Hard）すると良い。

2. **定量評価**:
   - 完走率 (Success Rate)
   - 衝突率 (Collision Rate)
   - 走行時間 (Lap Time)
   - 制御の滑らかさ (Jerk, Control Effort)

## 今後の拡張 (Future Work)
- **DAgger (Dataset Aggregation)**: Pure Pursuitのデータだけでは分布シフト（Distribution Shift）に弱いため、Tiny LiDAR Net自身が運転し、危ない場面でPure Pursuitが修正するようなInterventionデータを追加収集する。
