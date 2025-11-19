import math
import numpy as np

def build_obs(state, track):
    """
    Constructs the observation vector for the Neural Network.
    
    Args:
        state (VehicleState): Current vehicle state.
        track (Track): Track object.
        
    Returns:
        list: [e_lat, e_yaw, v, v_ref]
    """
    # Use the track's helper to get errors
    # Note: track.calculate_errors returns (cte, he)
    # We map cte -> e_lat, he -> e_yaw
    e_lat, e_yaw = track.calculate_errors(state.x, state.y, state.yaw)
    
    # Get v_ref at the nearest point
    ind = track.nearest_index(state.x, state.y)
    v_ref = track.v_ref[ind]
    
    return [e_lat, e_yaw, state.v, v_ref]
