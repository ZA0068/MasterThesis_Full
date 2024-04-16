import numpy as np
import matplotlib.pyplot as plt

class OIAC:
    def __init__(self, a=0.2, b=5, beta=0.1):
        self.a = a
        self.b = b
        self.beta = beta
        self.e = np.array([])
        self.e_dot = np.array([])
        self.epsilon = np.array([])
        self.tau_FF = np.array([0.0])
        self.u = np.array([])
        
    def calculate_error_position(self, q, qd):
        self.e = np.array(q) - np.array(qd)
    
    def set_saturation(self, lower, upper):
        self.lower = lower
        self.upper = upper
    
    def set_feedforward(self, array):
        self.tau_FF = array if isinstance(array, np.ndarray) else np.array([array])
    
    def calculate_error_velocity(self, q_dot, qd_dot):
        self.e_dot = np.array(q_dot) - np.array(qd_dot)
    
    def calculate_epsilon(self):
        self.epsilon = self.e + self.beta * self.e_dot
    
    def calculate_gamma(self):
        self.gamma = self.a / (1 + self.b * np.linalg.norm(self.epsilon)**2)
        
    def calculate_F(self):
        self.F = self.epsilon / self.gamma
    
    def calculate_K(self):
        self.K = self.F * self.e
    
    def calculate_B(self):
        self.D = self.F * self.e_dot
    
    def compute_output(self):
        self.u = - self.F - self.K * self.e - self.D * self.e_dot 
    
    def get_output(self):
        return (np.clip(self.u, self.lower, self.upper) if self.lower is not None and self.upper is not None else self.u) + self.tau_FF
    
    def run_OIAC(self, q: float, qd: float, q_dot: float, qd_dot: float):
        self.calculate_error_position(q, qd)
        self.calculate_error_velocity(q_dot, qd_dot)
        self.calculate_epsilon()
        self.calculate_gamma()
        self.calculate_F()
        self.calculate_K()
        self.calculate_B()
        self.compute_output()
        return self.get_output()
    
    def get_K(self):
        return self.K
    
    def get_D(self):
        return self.D
    
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

def main():
    oiac = OIAC(a=20, b=0.5, beta=0.05)
    oiac.set_saturation(-1, 10)
    
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
        out = oiac.run_OIAC(q, q_d, q_dot, q_d_dot)
        q += out
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