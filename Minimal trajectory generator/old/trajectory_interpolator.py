import numpy as np
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    def __init__(self, waypoints, durations):
        if len(waypoints) - 1 != len(durations):
            raise ValueError("Number of durations must be one less than the number of waypoints.")
        self.initialize_waypoints_and_durations(waypoints, durations)
        self.n = range(len(self.durations))

    def initialize_waypoints_and_durations(self, waypoints, durations):
        self.waypoints = np.array(waypoints, dtype=float)
        self.durations = np.array(durations, dtype=float)

    def smooth_pos(self, step):
        return 10 * step ** 3 - 15 * step ** 4 + 6 * step ** 5

    def smooth_vel(self, step):
        return 30 * step ** 2 - 60 * step ** 3 + 30 * step ** 4

    def smooth_acc(self, step):
        return 60 * step - 180 * step ** 2 + 120 * step ** 3

    def smooth_jrk(self, step):
        return - 360 * step + 360 * step ** 2

    def interpolate_segment(self, start_point, end_point, duration, num_points):
        time_samples = np.linspace(0, duration, num_points)
        interpolated_data = np.array([self.calculate_interpolation(start_point, end_point, duration, t) for t in time_samples])
        return interpolated_data[:,0,:], interpolated_data[:,1,:], interpolated_data[:,2,:], interpolated_data[:,3,:]

    def calculate_interpolation(self, start_point, end_point, duration, t):
        step = t / duration
        δ = end_point - start_point
        pos = start_point + δ * self.smooth_pos(step)
        vel = δ * self.smooth_vel(step) / duration
        acc = δ * self.smooth_acc(step) / (duration ** 2)
        jrk = δ * self.smooth_jrk(step) / (duration ** 3)
        return pos, vel, acc, jrk

    def generate(self, num_points_per_segment=100):
        all_positions, all_velocities, all_accelerations, all_jerks = [], [], [], []

        for i in self.n:
            segment_positions, segment_velocities, segment_accelerations, segment_jerks = self.interpolate_segment(
                self.waypoints[i], self.waypoints[i + 1], 
                self.durations[i], num_points_per_segment
            )
            all_positions.append(segment_positions)
            all_velocities.append(segment_velocities)
            all_accelerations.append(segment_accelerations)
            all_jerks.append(segment_jerks)
        self.data = (np.concatenate(all_positions), np.concatenate(all_velocities), 
                np.concatenate(all_accelerations), np.concatenate(all_jerks))
        return self.data

    def plot_data(self):
        positions, velocities, accelerations, jerks = self.data
        time = np.linspace(0, np.sum(self.durations), self.data[0].shape[0])
        dim = positions.shape[1]  # Dimension of the waypoints

        # Plotting based on dimensions
        if dim == 1:
            self.plot_1D(time, positions, velocities, accelerations, jerks)
        elif dim == 2:
            self.plot_2D(time, positions, velocities, accelerations, jerks)
        elif dim == 3:
            self.plot_3D(time, positions, velocities, accelerations, jerks)
        else:
            raise ValueError("Unsupported dimensionality for plotting")

    def plot_1D(self, time, positions, velocities, accelerations, jerks):
        plt.figure(figsize=(12, 8))
        plt.subplot(4, 1, 1)
        plt.plot(time, positions)
        plt.title("Position")
        plt.ylabel("Position")

        plt.subplot(4, 1, 2)
        plt.plot(time, velocities)
        plt.title("Velocity")
        plt.ylabel("Velocity")

        plt.subplot(4, 1, 3)
        plt.plot(time, accelerations)
        plt.title("Acceleration")
        plt.ylabel("Acceleration")

        plt.subplot(4, 1, 4)
        plt.plot(time, jerks)
        plt.title("Jerk")
        plt.ylabel("Jerk")
        plt.xlabel("Time")

        plt.tight_layout()
        plt.show()

    def plot_2D(self, time, positions, velocities, accelerations, jerks):
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))

        axs[0].plot(time, positions[:, 0], label='X')
        axs[0].plot(time, positions[:, 1], label='Y')
        axs[0].set_title('Position')
        axs[0].legend()

        axs[1].plot(time, velocities[:, 0], label='X')
        axs[1].plot(time, velocities[:, 1], label='Y')
        axs[1].set_title('Velocity')
        axs[1].legend()

        axs[2].plot(time, accelerations[:, 0], label='X')
        axs[2].plot(time, accelerations[:, 1], label='Y')
        axs[2].set_title('Acceleration')
        axs[2].legend()

        axs[3].plot(time, jerks[:, 0], label='X')
        axs[3].plot(time, jerks[:, 1], label='Y')
        axs[3].set_title('Jerk')
        axs[3].legend()

        _, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_title('Positions')
        ax.plot(positions[:, 0], positions[:, 1],  color='blue', label='Positions')
        ax.scatter(self.waypoints[0, 0], self.waypoints[0, 1], color='green', label='Start')
        ax.scatter(self.waypoints[1:-1, 0], self.waypoints[1:-1, 1], color='black', label='Waypoints')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_3D(self, time, positions, velocities, accelerations, jerks):
        fig = plt.figure(figsize=(12, 8))

        # 3D Trajectory Plot
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
        ax.set_title("3D Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Adding start and end points
        ax.scatter(*self.waypoints[0], color='green', label='Start Point')
        ax.scatter(*self.waypoints[-1], color='red', label='End Point')
        ax.legend()

        plt.tight_layout()
        plt.show()
        np.savetxt("straight_trajectory.csv", positions, delimiter=",")

if __name__ == '__main__':
    # Example usage with 2D waypoints
    waypoints = np.array([[2.48, -1.08, 1.0], [0.8, -2.0, 1.0], [-2.15, 0.03, 1.0], [-1.18, 1.03, 1.0], [0.75, 1.95, 1.0], [1.73, 1.03, 1.0], [0.8, 0.03, 1.0], [0.83, -0.95, 1.0], [2.48, -1.08, 1.0]])
    durations = np.array([5, 10, 4, 6, 3, 4, 2, 4])
    interpolator = TrajectoryGenerator(waypoints, durations)
    interpolator.generate()
    interpolator.plot_data()
