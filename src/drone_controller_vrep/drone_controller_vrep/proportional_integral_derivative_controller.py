import numpy as np
import matplotlib.pyplot as plt

class PID:
    def __init__(self, p=1, i=0, d=0):
        self.p = np.asarray(p)
        self.i = np.asarray(i)
        self.d = np.asarray(d)
        self.FF = np.asarray(0.0)
        self.lower = None
        self.upper = None
        self.prev_e = None
        self.cumsum = None
        
    def set_feedforward(self, feed_forward):
        self.FF = np.asarray(feed_forward)
        
    def set_saturation(self, lower, upper):
        self.lower = lower
        self.upper = upper
                
    def calculate_proportional(self):
        return self.p * self.e
    
    def calculate_integral(self):
        return self.i * self.cumsum
    
    def calculate_derivative(self):
        return self.d * (self.e - self.prev_e)
    
    def compute_output(self):
        self.out =  self.calculate_proportional() + self.calculate_integral() + self.calculate_derivative() 
        self.prev_e = self.e        
    
    
    def run_PID(self, q, qd, adt=None):
        self.calculate_error(q, qd)
        self.compute_output()
        self.apply_additional_terms(adt)
        return self.apply_saturation() + self.FF

    def apply_additional_terms(self, adt):
        if adt is not None:
            self.out += np.array(adt)

    def calculate_error(self, q, qd):
        self.e = np.array(qd) - np.array(q)
        self.accumulate_sum()
        if self.prev_e is None:
            self.prev_e = np.zeros_like(self.e, dtype=float)

    def accumulate_sum(self):
        if self.cumsum is not None:
            self.cumsum += self.e   
        else:
            self.cumsum = self.e
            
    def apply_saturation(self):
            if self.lower is not None and self.upper is not None:
                return np.clip(self.out, self.lower, self.upper) 
            return self.out
    
from scipy.interpolate import CubicSpline

def main():
    pid = PID(p=10, i=1, d=1)
    
    waypoints = np.array([[0.0, 0.0, 10.0], [10.0, 10.0, 10.0], [20.0, 15.0, 25.0], [30.0, 25.0, 10.0], [15.0, 50.0, 15.0]])
    durations = np.array([5, 5, 5, 5])
    
    # Create a time array from 0 to the sum of durations
    t = np.linspace(0, durations.sum(), 500)

    # Prepend a 0 to the durations.cumsum() array
    times = np.concatenate([[0], durations.cumsum()])

    # Create a cubic spline for each dimension
    x_spline = CubicSpline(times, waypoints[:, 0], bc_type='natural')
    y_spline = CubicSpline(times, waypoints[:, 1], bc_type='natural')
    z_spline = CubicSpline(times, waypoints[:, 2], bc_type='natural')

    # Calculate the desired trajectory and its derivative
    q_d_traj = np.vstack([x_spline(t), y_spline(t), z_spline(t)]).T
    q_d_dot_traj = np.vstack([x_spline(t, 1), y_spline(t, 1), z_spline(t, 1)]).T

    q = np.array([0.0, 0.0, 0.0])
    q_dot = np.array([0.0, 0.0, 0.0])
    
    qs = []
    for q_d, q_d_dot in zip(q_d_traj, q_d_dot_traj):
        out = pid.run_PID(q, q_d)
        q += out*0.1
        qs.append(q.copy())

    # Plotting the trajectory
    xs, ys, zs = zip(*qs)
    xd, yd, zd = zip(*q_d_traj)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, label='OIAC Trajectory')
    ax.plot(xd, yd, zd, label='Desired Trajectory')
    ax.scatter(*zip(*waypoints), color = 'red', marker = 'o', label = 'Waypoints')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()