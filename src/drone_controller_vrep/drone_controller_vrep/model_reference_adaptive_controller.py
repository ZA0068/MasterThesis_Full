import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline

class MRAC_PID:
    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.01, gamma=0.01):
        self.Kp = np.asarray(Kp)
        self.Ki = np.asarray(Ki)
        self.Kd = np.asarray(Kd)
        self.ff = None
        self.gamma = gamma
        self.wn = 1.0
        self.zeta = 0.8
        self.dt = 0.01
        self.integral_error = None
        self.derivative_error = None
        self.prev_error = None
        self.lower = None
        self.upper = None
        self.max_gain = 100.0
    
    def set_reference_model(self, wn, zeta):
        self.wn = wn
        self.zeta = zeta
    
    def set_max_gain(self, max_gain):
        self.max_gain = max_gain
    
    def set_dt(self, dt):
        self.dt = dt
    
    def set_feedforward(self, feedforward):
        self.ff = feedforward
    
    def set_adaptation(self, gamma):
        self.gamma = gamma
    
    def reference_model(self, y, r=1):
        y1, y2 = y
        return [y2, -2*self.zeta*self.wn*y2 - self.wn**2*y1 + self.wn**2*r]

    def calculate_proportional(self):
        return self.Kp * self.e
    
    def calculate_integral(self):
        self.accumulate_sum()
        return self.Ki * self.integral_error
    
    def calculate_derivative(self):
        self.evaluate_derivative_error()
        return self.Kd * self.derivative_error

    def evaluate_derivative_error(self):
        if self.prev_error is None:
            self.prev_error = np.zeros_like(self.e, dtype=float)
        self.derivative_error = (self.e - self.prev_error)
    
    def compute_output(self):
        self.out = self.calculate_proportional() + self.calculate_integral() + self.calculate_derivative()
        self.update_gains()
        self.prev_error = self.e
    
    def run_MRAC_PID(self, q, qd, ff=0):
        self.calculate_error(q, qd)
        self.compute_output()
        self.apply_saturation()
        return self.get_output(ff)

    def apply_saturation(self):
            if self.lower is not None and self.upper is not None:
                self.out = np.clip(self.out, self.lower, self.upper) 

    def get_output(self, ff=0):
        if self.ff is None:
            self.ff = np.zeros_like(self.e)
        return self.out + ff + self.ff

    def set_saturation(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def calculate_error(self, q, qd):
        self.e = np.asarray(qd, dtype=float) - np.asarray(q, dtype=float)

    def accumulate_sum(self):
        if self.integral_error is None:
            self.integral_error = np.zeros_like(self.e)
        self.integral_error = np.clip(self.integral_error + self.e * self.dt, -self.max_gain, self.max_gain)

    def update_gains(self):
        self.Kp = np.clip(self.Kp + self.gamma * self.e ** 2, -self.max_gain, self.max_gain)
        self.Ki = np.clip(self.Ki + self.gamma * self.e * self.integral_error, -self.max_gain, self.max_gain)
        self.Kd = np.clip(self.Kd + self.gamma * self.e * self.derivative_error, -self.max_gain, self.max_gain)

def main():
    mrac_pid = MRAC_PID(Kp=20.0, Ki=0.1, Kd=0.1, gamma=0.001)

    waypoints = np.array([[0.0, 0.0, 10.0], [10.0, 10.0, 10.0], [20.0, 15.0, 25.0], [30.0, 25.0, 10.0], [15.0, 50.0, 15.0]])
    durations = np.array([5, 5, 5, 5])

    t = np.arange(0, durations.sum(), 0.01)

    times = np.concatenate([[0], durations.cumsum()])

    x_spline = CubicSpline(times, waypoints[:, 0], bc_type='natural')
    y_spline = CubicSpline(times, waypoints[:, 1], bc_type='natural')
    z_spline = CubicSpline(times, waypoints[:, 2], bc_type='natural')

    q_d_traj = np.vstack([x_spline(t), y_spline(t), z_spline(t)]).T

    q = np.array([0.0, 0.0, 0.0])
    qs = []
    mrac_pid.set_dt(0.01)
    for i in range(1, len(t)):
        q_d = q_d_traj[i]
        out = mrac_pid.run_MRAC_PID(q, q_d)
        q += out * 0.01
        qs.append(q.copy())

    # Plotting the trajectory
    qs = np.array(qs)
    xs, ys, zs = qs[:, 0], qs[:, 1], qs[:, 2]
    xd, yd, zd = q_d_traj[:, 0], q_d_traj[:, 1], q_d_traj[:, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, label='MRAC Trajectory')
    ax.plot(xd, yd, zd, label='Desired Trajectory')
    ax.scatter(*zip(*waypoints), color='red', marker='o', label='Waypoints')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
