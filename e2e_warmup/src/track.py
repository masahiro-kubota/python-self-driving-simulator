import pandas as pd
import numpy as np
import math

class Track:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.x = self.df['x'].values
        self.y = self.df['y'].values
        
        if 'yaw' in self.df.columns:
            self.yaw = self.df['yaw'].values
        elif 'z_quat' in self.df.columns and 'w_quat' in self.df.columns:
            # Calculate yaw from quaternion (assuming z-axis rotation)
            # yaw = atan2(2(w*z + x*y), 1 - 2(y^2 + z^2))
            # Simplified for 2D (x=0, y=0 for pure z-rotation, but let's use full formula to be safe or just z/w if others are 0)
            # The provided CSV has x_quat=0, y_quat=0.
            # Yaw = atan2(2*w*z, 1 - 2*z^2)
            z = self.df['z_quat'].values
            w = self.df['w_quat'].values
            self.yaw = np.arctan2(2.0 * w * z, 1.0 - 2.0 * z * z)
        else:
            # Default to 0 or calculate from xy
            self.yaw = np.zeros_like(self.x)
            
        if 'v_ref' in self.df.columns:
            self.v_ref = self.df['v_ref'].values
        elif 'speed' in self.df.columns:
            self.v_ref = self.df['speed'].values
        else:
            self.v_ref = np.zeros_like(self.x)
        
    def nearest_index(self, x, y):
        """
        Finds the index of the nearest point on the track to the given (x, y).
        """
        dx = self.x - x
        dy = self.y - y
        dist = np.hypot(dx, dy)
        ind = np.argmin(dist)
        return ind

    def calculate_errors(self, x, y, yaw):
        """
        Calculates Cross Track Error (CTE) and Heading Error (HE).
        
        Args:
            x, y, yaw: Vehicle state.
            
        Returns:
            cte: Cross Track Error [m].
            he: Heading Error [rad].
        """
        ind = self.nearest_index(x, y)
        
        # Track point and yaw
        tx = self.x[ind]
        ty = self.y[ind]
        tyaw = self.yaw[ind]
        
        # CTE
        dx = x - tx
        dy = y - ty
        # Cross product to determine sign: (tx_vec) x (d_vec)
        # Track vector: [cos(tyaw), sin(tyaw)]
        # d_vec: [dx, dy]
        # cross = cos(tyaw)*dy - sin(tyaw)*dx
        cte = np.hypot(dx, dy)
        cross = math.cos(tyaw) * dy - math.sin(tyaw) * dx
        if cross < 0:
            cte = -cte
            
        # HE
        he = tyaw - yaw
        # Normalize to -pi to pi
        he = (he + math.pi) % (2 * math.pi) - math.pi
        
        return cte, he
