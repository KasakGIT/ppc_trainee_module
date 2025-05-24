import numpy as np

class PIDController:
    def __init__(self, Kp=1.0, Ki=0.2, Kd=0.7):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.previous_error = 0.0
         
        
    def update(self, error, dt):
        # Proportional term
        P = self.Kp * error
        
        # Integral term 
        self.integral += error * dt
        I = self.Ki * self.integral
        
        # Derivative term
        D = 0.0
        if dt > 1e-6:  # Avoid division by zero
            D = self.Kd * (error - self.previous_error) / dt
        
        output = P + I + D
        
        self.previous_error = error
        return output

def stanley_steering(x, y, yaw, v, waypoints, k=5.0, ks=1e-2, max_steer=np.radians(30)):
    # Front axle position
    L = 2.5  # wheelbase
    fx = x + L * np.cos(yaw)
    fy = y + L * np.sin(yaw)
    
    # Find nearest waypoint 
    target_idx = np.argmin(np.sum((waypoints - np.array([fx, fy]))**2, axis=1))
    
    # Calculate path heading 
    if target_idx < len(waypoints) - 1:
        dx, dy = waypoints[target_idx+1] - waypoints[target_idx]
    else:
        dx, dy = waypoints[target_idx] - waypoints[target_idx-1]
    path_yaw = np.arctan2(dy, dx)
    
    # Normalized heading error [-π, π]
    heading_error = (path_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
    
    # Cross-track error (signed distance)
    path_normal = np.array([-np.sin(path_yaw), np.cos(path_yaw)])
    cross_track_error = np.dot([fx - waypoints[target_idx,0], fy - waypoints[target_idx,1]], path_normal)
    
    # Stanley control law
    cross_track_steer = np.arctan2(k * cross_track_error, max(v, 0.1) + ks)  # Ensure non-zero velocity
    steer = heading_error + cross_track_steer
    steer = np.clip(steer, -max_steer, max_steer)
    
    return steer, target_idx

pid_controller = PIDController(Kp=0.5, Ki=0.05, Kd=0.3)

def pid_throttle(error, dt):
    """Wrapper for PID throttle control"""
    return pid_controller.update(error, dt)
