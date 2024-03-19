# Let's write the full code to calculate the Bezier curve with minimal jerk trajectory, ensuring that the
# endpoint does not connect back to the startpoint.

import numpy as np
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    def __init__(self, waypoints, durations):
        if len(waypoints) - 1 != len(durations):
            raise ValueError("Number of durations must be one less than the number of waypoints.")
        
        self.waypoints = waypoints if isinstance(waypoints, np.ndarray) else np.array(waypoints, dtype=float)
        self.durations = durations if isinstance(durations, np.ndarray) else np.array(durations, dtype=float)
        
        self.dimensions = self.waypoints.shape[1]
        self.number_of_segments = len(self.durations)
        self.total_duration = np.sum(self.durations)
        self.control_points = np.array([])
        
        self.set_resolution(100)
        
    def smooth_pos(self, step):
        return 10 * step ** 3 - 15 * step ** 4 + 6 * step ** 5

    def smooth_vel(self, step):
        return 30 * step ** 2 - 60 * step ** 3 + 30 * step ** 4

    def smooth_acc(self, step):
        return 60 * step - 180 * step ** 2 + 120 * step ** 3

    def smooth_jrk(self, step):
        return - 360 * step + 360 * step ** 2
    
    def set_resolution(self, number_of_points_per_segment):
        self.number_of_points_per_segment = number_of_points_per_segment
        self.total_number_of_points = self.number_of_segments * self.number_of_points_per_segment
        self.full_t_values = np.linspace(0, self.total_duration, self.total_number_of_points)
        self.position_points = np.zeros((self.total_number_of_points, self.dimensions))
        self.velocity_points = np.zeros((self.total_number_of_points, self.dimensions))
        self.acceleration_points = np.zeros((self.total_number_of_points, self.dimensions))
        self.jerk_points = np.zeros((self.total_number_of_points, self.dimensions))

    
    # Define a function to calculate the Bezier curve points for a segment
    def calculate_bezier_segment(self, p0, p1, p2, p3, t_values, derivative = 0):
        T = self.get_T_vector(t_values, derivative)
        M = np.array([[-1, 3, -3, 1], 
                      [3, -6, 3, 0], 
                      [-3, 3, 0, 0], 
                      [1, 0, 0, 0]])
        P = np.array([p0, p1, p2, p3])
        return T @ M @ P

    def get_T_vector(self, t_values, derivative):
        match derivative:
            case 0:
                return np.array([t_values**3, t_values**2, t_values, np.ones_like(t_values)]).T
            case 1:
                return np.array([3*t_values**2, 2*t_values, np.ones_like(t_values), np.zeros_like(t_values)]).T
            case 2:
                return np.array([6*t_values, 2*np.ones_like(t_values), np.zeros_like(t_values), np.zeros_like(t_values)]).T
            case 3:
                return np.array([6*np.ones_like(t_values), np.zeros_like(t_values), np.zeros_like(t_values), np.zeros_like(t_values)]).T
            case _:
                raise ValueError("Invalid derivative. Must be 0 (Position), 1 (Velocity), 2 (Acceleration), or 3 (Jerk).")

    # Calculate the full Bezier curve with minimal jerk across all segments
    def calculate_full_bezier_curve(self):
        for i in range(self.number_of_segments):
            # Calculate segment time values
            segment_t_values = np.linspace(0, self.durations[i], self.number_of_points_per_segment) / self.durations[i]

            # Apply minimal jerk to time values
            t_values_for_position       = self.smooth_pos(segment_t_values)
            t_values_for_velocity       = self.smooth_vel(segment_t_values) / self.durations[i]
            t_values_for_acceleration   = self.smooth_acc(segment_t_values) / self.durations[i]**2
            t_values_for_jerk           = self.smooth_jrk(segment_t_values) / self.durations[i]**3
            
            # Calculate curve points for segment
            p0, p1, p2, p3 = self.get_local_points(i)
            min_index = self.number_of_points_per_segment * i
            max_index = self.number_of_points_per_segment * (i+1)
            self.position_points[min_index:max_index]       = self.calculate_bezier_segment(p0, p1, p2, p3, t_values_for_position, derivative=0)
            self.velocity_points[min_index:max_index]       = self.calculate_bezier_segment(p0, p1, p2, p3, t_values_for_velocity, derivative=1)
            self.acceleration_points[min_index:max_index]   = self.calculate_bezier_segment(p0, p1, p2, p3, t_values_for_acceleration, derivative=2)
            self.jerk_points[min_index:max_index]           = self.calculate_bezier_segment(p0, p1, p2, p3, t_values_for_jerk, derivative=3)
            self.data_points = (self.position_points, self.velocity_points, self.acceleration_points, self.jerk_points)

    def get_local_points(self, i):
        return self.waypoints[i], self.control_points[2*i], self.control_points[2*i+1], self.waypoints[i+1]

    # Function to calculate control points for each segment
    def calculate_control_points(self):
        # First and last control points
        control_points = [self.waypoints[0] + (self.waypoints[1] - self.waypoints[0]) / 5]

        for i in range(1, len(self.waypoints) - 1):
            # Vector from previous to current and next to current
            prev_to_current = self.waypoints[i] - self.waypoints[i - 1]
            next_to_current = self.waypoints[i + 1] - self.waypoints[i]

            # Normalize to get direction
            direction_prev = prev_to_current / np.linalg.norm(prev_to_current)
            direction_next = next_to_current / np.linalg.norm(next_to_current)

            # Average direction for C1 continuity (tangency)
            avg_direction = (direction_prev + direction_next) / 2

            # Set control points along the average direction
            control_points.append(self.waypoints[i] - avg_direction * np.linalg.norm(prev_to_current) / 3)
            control_points.append(self.waypoints[i] + avg_direction * np.linalg.norm(next_to_current) / 3)

        control_points.append(self.waypoints[-1] - (self.waypoints[-1] - self.waypoints[-2]) / 3)
        
        # Convert to numpy array as member variable
        self.control_points = np.array(control_points)

    def plot_positions(self):
        titles = ['Position', 'Velocity', 'Acceleration', 'Jerk']
        units = ['[m]', '[m/s]', '[m/s^2]', '[m/s^3]']
        labels = ['X', 'Y', 'Z'][:self.dimensions]
        _, axs = plt.subplots(2, 2, figsize=(12, 12))
        for i in range(4):
            ax = axs[i // 2, i % 2]  # Determine the correct subplot
            for j in range(self.dimensions):
                ax.plot(self.full_t_values, self.data_points[i][:, j], label=f'{labels[j]} {titles[i]}')
            ax.set_title(f'{titles[i]} Over Time with Minimal Jerk Trajectory')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(titles[i] + ' ' + units[i])
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trajectory(self):
        titles = ['Position', 'Velocity', 'Acceleration', 'Jerk']
        labels = ['X', 'Y', 'Z'][:self.dimensions]
        fig = plt.figure(figsize=(12, 12))

        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1, projection='3d' if self.dimensions == 3 else None)
            ax.plot(*[self.data_points[i][:, j] for j in range(self.dimensions)], label=f'{titles[i]} curve with Minimal Jerk')
            ax.set_title(titles[i])
            for j, label in enumerate(labels):
                getattr(ax, f'set_{label.lower()}label')(f'{label} {titles[i]} [m]')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_data(self):
        self.plot_trajectory()
        self.plot_positions()

# Define waypoints and durations
waypoints = np.array([[0, 0], [10, 20], [20, 10], [30, 30], [0, 0]], dtype=float)
#waypoints = np.array([[0, 0, 10], [10, 20, 15], [20, 10, 15], [30, 30, 30], [0, 0, 10]], dtype=float)
durations = np.array([5, 4, 7, 5])
trajectory_generator = TrajectoryGenerator(waypoints=waypoints, durations=durations)

# Calculate control points for the Bezier curve
trajectory_generator.calculate_control_points()
trajectory_generator.calculate_full_bezier_curve()
trajectory_generator.plot_data()
