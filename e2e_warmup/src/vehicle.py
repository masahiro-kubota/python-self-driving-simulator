import math

class VehicleState:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def step(state, delta, a, dt, L=2.5):
    """
    Updates the vehicle state using the kinematic bicycle model.
    
    Args:
        state (VehicleState): Current state.
        delta (float): Steering angle [rad].
        a (float): Acceleration [m/s^2].
        dt (float): Time step [s].
        L (float): Wheelbase [m].
        
    Returns:
        VehicleState: Updated state (new instance).
    """
    
    # Kinematic bicycle model equations
    # x_dot = v * cos(yaw)
    # y_dot = v * sin(yaw)
    # yaw_dot = v / L * tan(delta)
    # v_dot = a
    
    x_next = state.x + state.v * math.cos(state.yaw) * dt
    y_next = state.y + state.v * math.sin(state.yaw) * dt
    yaw_next = state.yaw + state.v / L * math.tan(delta) * dt
    v_next = state.v + a * dt
    
    # Normalize yaw to -pi to pi (optional but good practice)
    yaw_next = (yaw_next + math.pi) % (2 * math.pi) - math.pi
    
    return VehicleState(x_next, y_next, yaw_next, v_next)
