import numpy as np
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    def __init__(self, waypoints, durations):
        if len(waypoints) - 1 != len(durations):
            raise ValueError("Number of durations must be one less than the number of waypoints.")
        self.initialize_waypoints_and_durations(waypoints, durations)
        self.n = range(len(self.durations))

    def initialize_waypoints_and_durations(self, waypoints, durations):
        if isinstance(waypoints, np.ndarray):
            self.waypoints = waypoints
        else:
            self.waypoints = np.array(waypoints)
        if isinstance(durations, np.ndarray):
            self.durations = durations
        else:
            self.durations = np.array(durations)

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
        positions, velocities, accelerations, jerks = [], [], [], []

        for t in time_samples:
            pos, vel, acc, jrk = self.calculate_interpolation(start_point, end_point, duration, t)
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
            jerks.append(jrk)

        return np.array(positions), np.array(velocities), np.array(accelerations), np.array(jerks)

    def calculate_interpolation(self, start_point, end_point, duration, t):
        step = t / duration
        δ = end_point - start_point
        pos = start_point + δ * self.smooth_pos(step)
        vel = δ * self.smooth_vel(step) / (duration)
        acc = δ * self.smooth_acc(step) / (duration ** 2)
        jrk = δ * self.smooth_jrk(step) / (duration ** 3)
        return pos,vel,acc,jrk

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
            # Plotting

        # Position Plot
        _, axs1 = plt.subplots(1, 1, figsize=(10, 4))
        axs1.plot(time, positions)
        axs1.set_title("Position plot")
        axs1.set_xlabel("Time (s)")
        axs1.set_ylabel("Position")

        # Velocity Plot
        _, axs2 = plt.subplots(1, 1, figsize=(10, 4))
        axs2.plot(time, velocities)
        axs2.set_title("Velocity plot")
        axs2.set_xlabel("Time (s)")
        axs2.set_ylabel("Velocity")

        # Acceleration Plot
        _, axs3 = plt.subplots(1, 1, figsize=(10, 4))
        axs3.plot(time, accelerations)
        axs3.set_title("Acceleration plot")
        axs3.set_xlabel("Time (s)")
        axs3.set_ylabel("Acceleration")

        # Acceleration Plot
        _, axs4 = plt.subplots(1, 1, figsize=(10, 4))
        axs4.plot(time, jerks)
        axs4.set_title("Jerk plot")
        axs4.set_xlabel("Time (s)")
        axs4.set_ylabel("Jerk")

        # Plotting the trajectory in 3D space (XYZ)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extracting X, Y, Z coordinates from the positions array
        X = positions[:, 0]
        Y = positions[:, 1]
        Z = positions[:, 2]

        # Plotting the trajectory
        ax.plot(X, Y, Z, label='3D Trajectory')

        # Setting labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title('3D Trajectory Interpolation')

        # Adding start and end points
        ax.scatter(*waypoints[0], color='green', label='Start Point')
        ax.scatter(*waypoints[-1], color='red', label='End Point')

        # Legend
        ax.legend()

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Example usage
    waypoints = [[0, 0, 10], [10, 10, 10], [20, 15, 25], [30, 25, 10], [0, 0, 10]]
    durations = [5, 10, 15, 20]
    interpolator = TrajectoryGenerator(waypoints, durations)
    interpolator.generate()
    interpolator.plot_data()
    


