"""Geometry utility functions."""

import numpy as np


def normalize_angle(angle: float) -> float:
    """角度を[-π, π]に正規化.

    Args:
        angle: 角度 [rad]

    Returns:
        正規化された角度 [rad]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """2点間のユークリッド距離を計算.

    Args:
        x1, y1: 点1の座標
        x2, y2: 点2の座標

    Returns:
        距離 [m]
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def angle_between_points(x1: float, y1: float, x2: float, y2: float) -> float:
    """2点間の角度を計算.

    Args:
        x1, y1: 点1の座標
        x2, y2: 点2の座標

    Returns:
        角度 [rad]
    """
    return np.arctan2(y2 - y1, x2 - x1)


def rotate_point(
    x: float,
    y: float,
    angle: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> tuple[float, float]:
    """点を原点周りに回転.

    Args:
        x, y: 点の座標
        angle: 回転角度 [rad]
        origin_x, origin_y: 回転中心の座標

    Returns:
        回転後の座標 (x, y)
    """
    # 原点に平行移動
    x_shifted = x - origin_x
    y_shifted = y - origin_y

    # 回転
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_rotated = x_shifted * cos_angle - y_shifted * sin_angle
    y_rotated = x_shifted * sin_angle + y_shifted * cos_angle

    # 元の位置に戻す
    return x_rotated + origin_x, y_rotated + origin_y


def nearest_point_on_line(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float, float]:
    """点から線分への最近点を計算.

    Args:
        px, py: 点の座標
        x1, y1: 線分の始点
        x2, y2: 線分の終点

    Returns:
        最近点の座標 (x, y) と距離
    """
    # 線分のベクトル
    dx = x2 - x1
    dy = y2 - y1

    # 線分の長さの2乗
    length_sq = dx * dx + dy * dy

    if length_sq == 0:
        # 線分が点の場合
        return x1, y1, distance(px, py, x1, y1)

    # 点から線分への射影パラメータ
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))

    # 最近点
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    # 距離
    dist = distance(px, py, nearest_x, nearest_y)

    return nearest_x, nearest_y, dist


def curvature_from_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> float:
    """3点から曲率を計算（外接円の半径の逆数）.

    Args:
        x1, y1: 点1の座標
        x2, y2: 点2の座標
        x3, y3: 点3の座標

    Returns:
        曲率 [1/m]
    """
    # 3点が一直線上にある場合
    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    if area < 1e-10:
        return 0.0

    # 辺の長さ
    a = distance(x2, y2, x3, y3)
    b = distance(x1, y1, x3, y3)
    c = distance(x1, y1, x2, y2)

    # 外接円の半径
    radius = (a * b * c) / (4.0 * area)

    # 曲率
    return 1.0 / radius if radius > 0 else 0.0


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """オイラー角からクォータニオン(x, y, z, w)に変換.

    Args:
        roll: ロール角 [rad]
        pitch: ピッチ角 [rad]
        yaw: ヨー角 [rad]

    Returns:
        クォータニオン (x, y, z, w)
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w


__all__ = [
    "angle_between_points",
    "curvature_from_points",
    "distance",
    "euler_to_quaternion",
    "nearest_point_on_line",
    "normalize_angle",
    "rotate_point",
]
