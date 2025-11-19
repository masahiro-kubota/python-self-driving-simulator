import math
import numpy as np

def pure_pursuit_control(state, track, L, lookahead_dist):
    """
    Calculates the steering angle using Pure Pursuit algorithm.
    
    Args:
        state (VehicleState): Current vehicle state.
        track (Track): Track object.
        L (float): Wheelbase [m].
        lookahead_dist (float): Lookahead distance [m].
        
    Returns:
        delta (float): Steering angle [rad].
        target_ind (int): Index of the target point on the track.
    """
    
    # 1. Find nearest point index
    # Optimization: search around the previous index if available (not implemented here for simplicity, 
    # assuming track.nearest_index is fast enough or we just search globally)
    nearest_ind = track.nearest_index(state.x, state.y)
    
    # 2. Search for target point
    # We want a point that is lookahead_dist away from the *current vehicle position*?
    # Or from the nearest point? Standard PP usually looks for intersection with lookahead circle.
    # A simplified version searches forward from nearest_ind until distance > lookahead_dist.
    
    target_ind = nearest_ind
    distance = 0.0
    
    # Search forward
    while True:
        # Calculate distance from vehicle to this point
        dx = track.x[target_ind] - state.x
        dy = track.y[target_ind] - state.y
        dist_to_point = math.hypot(dx, dy)
        
        if dist_to_point >= lookahead_dist:
            break
        
        target_ind = (target_ind + 1) % len(track.x)
        
        # Safety break if we looped around and didn't find a point (track too small?)
        if target_ind == nearest_ind:
            break
            
    # 3. Calculate alpha
    tx = track.x[target_ind]
    ty = track.y[target_ind]
    
    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw
    
    # 4. Calculate delta
    # delta = atan2(2 * L * sin(alpha) / Ld)
    delta = math.atan2(2.0 * L * math.sin(alpha), lookahead_dist)
    
    return delta, target_ind

def pid_control(target_v, current_v, Kp):
    return Kp * (target_v - current_v)
