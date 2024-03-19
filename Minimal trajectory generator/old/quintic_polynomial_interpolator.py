import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class ExtendedMinimalJerkTrajectory:
    def __init__(self, waypoints, durations):
        self.waypoints = np.array(waypoints)
        self.durations = np.array(durations)
        self.total_time = np.sum(self.durations)
        self.dimension = self.waypoints.shape[1]
        self.cs = [CubicSpline(np.arange(len(self.waypoints[:, i])), self.waypoints[:, i]) for i in range(self.dimension)]

    def minimal_jerk_tau(self, t, duration):
        tau = t / duration
        return 6 * tau**5 - 15 * tau**4 + 10 * tau**3

    def evaluate(self, t, derivative=0):
        if t < 0 or t > self.total_time:
            raise ValueError("Time t is out of bounds.")

        elapsed_time = 0
        for i, duration in enumerate(self.durations):
            if elapsed_time + duration >= t:
                tau = self.minimal_jerk_tau(t - elapsed_time, duration)
                return np.array([spline(i + tau, nu=derivative) for spline in self.cs])
            elapsed_time += duration

        raise ValueError("Time t is out of range.")

    def plot_trajectory(self, plot_3d=False):
        t_samples = np.linspace(0, self.total_time, 1000)
        trajectory = np.array([self.evaluate(t) for t in t_samples])

        if self.dimension == 3 and plot_3d:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Minimal Jerk Trajectory')
            ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], color='red', label='Waypoints')
            ax.set_title('3D Minimal Jerk Trajectory')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position')
        else:
            plt.figure(figsize=(10, 8))
            plt.plot(trajectory[:, 0], trajectory[:, 1], label='Minimal Jerk Trajectory')
            plt.scatter(self.waypoints[:, 0], self.waypoints[:, 1], color='red', label='Waypoints')
            plt.title('Minimal Jerk Trajectory')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')

        plt.legend()
        plt.grid(True)
        plt.axis('auto')
        plt.show()

    def plot_property_over_time(self, property_index):
        properties = ['Position', 'Velocity', 'Acceleration', 'Jerk']
        if property_index not in range(4):
            raise ValueError("Invalid property index. Must be 0 (Position), 1 (Velocity), 2 (Acceleration), or 3 (Jerk).")

        t_samples = np.linspace(0, self.total_time, 1000)
        property_values = np.array([self.evaluate(t, derivative=property_index) for t in t_samples])

        plt.figure(figsize=(10, 8))
        for i in range(self.dimension):
            plt.plot(t_samples, property_values[:, i], label=f'{properties[property_index]} in {chr(88+i)}')

        plt.title(f'{properties[property_index]} Over Time')
        plt.xlabel('Time')
        plt.ylabel(properties[property_index])
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
waypoints = [[0, 0], [10, 20], [20, 10], [30, 30], [0, 0]]
durations = [5, 10, 15, 20]
trajectory = ExtendedMinimalJerkTrajectory(waypoints, durations)
trajectory.plot_trajectory(plot_3d=True)
trajectory.plot_property_over_time(0)  # Plotting velocity
trajectory.plot_property_over_time(1)  # Plotting velocity
trajectory.plot_property_over_time(2)  # Plotting acceleration
trajectory.plot_property_over_time(3)  # Plotting jerk

