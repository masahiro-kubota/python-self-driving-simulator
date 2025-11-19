import numpy as np
import pandas as pd
import math

def generate_track():
    # Track parameters
    straight_len = 50.0
    radius = 20.0
    resolution = 0.5  # meters between points
    v_ref = 8.33  # ~30 km/h

    points = []

    # 1. Straight (0,0) to (straight_len, 0)
    x = np.arange(0, straight_len, resolution)
    y = np.zeros_like(x)
    yaw = np.zeros_like(x)
    for i in range(len(x)):
        points.append([x[i], y[i], 0.0, yaw[i], v_ref])

    # 2. Turn 1 (180 degrees)
    # Center of turn is (straight_len, radius)
    # Start angle -90 deg (pointing right), End angle 90 deg (pointing left) -> No, wait.
    # Current heading is 0 (East). We want to turn Left (Counter Clockwise) to head West.
    # So we turn 180 degrees.
    # Center of circle: (straight_len, radius)
    # Entry point: (straight_len, 0) -> Angle relative to center: -90 deg (3*pi/2)
    # Exit point: (straight_len, 2*radius) -> Angle relative to center: 90 deg (pi/2)
    
    center_x1 = straight_len
    center_y1 = radius
    
    # Angles from -pi/2 to pi/2
    angles1 = np.arange(-np.pi/2, np.pi/2, resolution/radius)
    for theta in angles1:
        px = center_x1 + radius * np.cos(theta)
        py = center_y1 + radius * np.sin(theta)
        # Tangent angle: theta + pi/2
        pyaw = theta + np.pi/2
        # Normalize yaw to -pi to pi
        pyaw = (pyaw + np.pi) % (2 * np.pi) - np.pi
        points.append([px, py, 0.0, pyaw, v_ref])

    # 3. Straight back (straight_len, 2*radius) to (0, 2*radius)
    # Heading West (pi or -pi)
    # x goes from straight_len down to 0
    x_back = np.arange(straight_len, 0, -resolution)
    y_back = np.full_like(x_back, 2*radius)
    yaw_back = np.full_like(x_back, np.pi)
    for i in range(len(x_back)):
        points.append([x_back[i], y_back[i], 0.0, yaw_back[i], v_ref])

    # 4. Turn 2 (180 degrees)
    # Center of turn is (0, radius)
    # Entry point: (0, 2*radius) -> Angle relative to center: pi/2
    # Exit point: (0, 0) -> Angle relative to center: 3*pi/2 (-pi/2)
    
    center_x2 = 0
    center_y2 = radius
    
    # Angles from pi/2 to 3*pi/2
    angles2 = np.arange(np.pi/2, 3*np.pi/2, resolution/radius)
    for theta in angles2:
        px = center_x2 + radius * np.cos(theta)
        py = center_y2 + radius * np.sin(theta)
        # Tangent angle: theta + pi/2
        pyaw = theta + np.pi/2
        # Normalize yaw
        pyaw = (pyaw + np.pi) % (2 * np.pi) - np.pi
        points.append([px, py, 0.0, pyaw, v_ref])

    # Create DataFrame
    df = pd.DataFrame(points, columns=['x', 'y', 'z', 'yaw', 'v_ref'])
    
    # Save to CSV
    output_path = 'data/tracks/track.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} points.")

if __name__ == "__main__":
    generate_track()
