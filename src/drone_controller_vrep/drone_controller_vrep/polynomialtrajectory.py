import itertools
from header_file import *
from polynomialgenerator import PolynomialGenerator
class MinimalTrajectoryGenerator(PolynomialGenerator):
    def __init__(self, waypoints=None, durations=None, minimal_trajectory_derivative=None):
        self.reset()
        if any([waypoints is not None, durations is not None, minimal_trajectory_derivative is not None]):
            self.initialize(waypoints, durations, minimal_trajectory_derivative)

    def initialize(self, waypoints, durations, minimal_trajectory_derivative):
        self.initialize_spline_parameters(waypoints, durations)
        self.initialize_polynomial_parameters(minimal_trajectory_derivative)
        
        
    def reset(self):
        self.__number_of_waypoints = 0
        self.__number_of_splines = 0
        self.__dimensions = 0
        self.__splines = 0
        self._degrees = None
        self._degrees = 0
        self.__nth_row = 0
        self.__number_of_coefficients = 0
        self.__A = None
        self.__b = None
        self.__maximum_velocity = 1.0
        self.__dt = 0.05 
        
    def set_maximum_velocity(self, max_velocity):
        self.__maximum_velocity = max_velocity
        
    def set_dt(self, dt):
        self.__dt = dt

    def set_start_point(self, start_point):
        self.__waypoints[0] = start_point
        if self.__b is not None:
            self.__b[0] = start_point
        
    def set_end_point(self, end_point):
        self.__waypoints[-1] = end_point
        if self.__b is not None:
            self.__b[self.__number_of_waypoints] = end_point

    def initialize_polynomial_parameters(self, minimal_derivative):
        self.set_degrees(diff2deg(minimal_derivative))
        self.set_derivative(minimal_derivative)
        self.set_max_continuity_derivative()
        self.set_minmax_start_and_goal_derivative()
        self.__number_of_coefficients = self._degrees + 1

    def initialize_spline_parameters(self, waypoints, duration):
        if waypoints is None:
            raise ValueError("Waypoints must be provided!!!")
        self.set_waypoints(waypoints)
        self.set_durations(duration)
        self.validate_input_data()
        self.__splines = np.array([])

    def set_durations(self, duration):
        self.__durations = None if duration is None else ensure_numpy_array(duration)

    def _instantiate_durations(self):
        if self.__durations is None:
            self.__durations = np.linalg.norm(np.diff(self.__waypoints, axis=0), axis=1) / self.__maximum_velocity
        else:
            self.set_time_based_on_velocity()
        self.__number_of_splines = self.__durations.shape[0]
        self.__time_matrix = np.hstack((0, np.cumsum(self.__durations)))
        self.__dur_mat = np.zeros(self.__durations.size * 2 + 1, dtype=self.__durations.dtype)
        self.__dur_mat[1::2] = self.__durations

    def set_waypoints(self, positions):
        self.__waypoints = ensure_numpy_array(positions)

    def _instantiate_waypoints(self):
        self.__number_of_waypoints = np.max(self.__waypoints.shape[0] - 2, 0)
        self.__dimensions = self.__waypoints.shape[1]

    def validate_input_data(self):
        if self.__durations is None:
            return
        if self.__waypoints.shape[0] - 1 != self.__durations.shape[0]:
            raise ValueError(f"Number of waypoints and duration must match!!!\n Number of waypoints: {self.__waypoints.shape[0]}\n Number of durations: {self.__durations.shape[0]}")
        
    def generate_nth_degrees(self, trajectory_type):
        if trajectory_type.__class__.__name__ == 'Degree':
            return trajectory_type
        return diff2deg(trajectory_type)
    
    def set_maximum_velocity(self, max_vel):
        self.__maximum_velocity = max_vel
    
    def set_dt(self, dt):
        self.__dt = dt
    
    def set_start_point(self, start_point):
        self.__waypoints[0] = start_point
        if self.__b is not None:
            self.__b[0] = start_point
    
    def set_end_point(self, end_point):
        self.__waypoints[-1] = end_point
        if self.__b is not None:
            self.__b[self.__number_of_waypoints] = end_point
        
    
    def init_AB_matrices(self):
        self.__A = np.zeros((self.__number_of_splines * self.__number_of_coefficients, self.__number_of_coefficients * self.__number_of_splines))
        self.__b = np.zeros((self.__number_of_splines * self.__number_of_coefficients, self.__dimensions))

    def create_poly_matrices(self):
        self._instantiate_waypoints()
        self._instantiate_durations()
        self.init_AB_matrices()
        self.generate_position_constraints()
        self.generate_continuity_constraints()
        self.generate_start_and_goal_constraints()

    def set_time_based_on_velocity(self):
        time = np.linalg.norm(np.diff(self.__waypoints, axis=0), axis=1) / self.__maximum_velocity
        self.__durations = np.maximum(time, self.__durations)

    def compute_splines(self):
        self.coefficients = np.linalg.lstsq(self.__A, self.__b, rcond=None)[0]
        result = [
            tuple(
                self.generate_polynomial(derivative=d, T=t) @ coefficients_slice
                for d in range(self._degrees+1)
            )
            for spline in range(self.__number_of_splines)
            for t in np.arange(0.0, self.__durations[spline], self.__dt)
            for coefficients_slice in [self.coefficients[spline * self.__number_of_coefficients: (spline + 1) * self.__number_of_coefficients]]
        ]
        self.__splines = np.hstack(list(zip(*result)))


    def get_splines(self):
        return self.__splines

    def set_max_continuity_derivative(self, max_derivative=-1):
        if max_derivative < 1:
            self.__maximum__continuity_derivative = self._degrees
        else:
            self.__maximum__continuity_derivative = np.clip(max_derivative+1, 2, self._degrees)
    
    def generate_continuity_constraints(self):
        for derivative in range(1, self.__maximum__continuity_derivative):
            for spline in range(1, self.__number_of_splines):
                polyT = self.generate_polynomial(derivative=derivative,T=self.__durations[spline-1])
                poly0 = self.generate_polynomial(derivative=derivative,T=0)
                self.__A[self.__nth_row, self._segment(spline-1):self._segment(spline+1)] = np.hstack((polyT, -poly0))
                self.__nth_row += 1

    def generate_position_constraints(self):
        for spline, offset in itertools.product(range(self.__number_of_splines), range(2)):
            poly = self.generate_polynomial(derivative=0, T=self.__dur_mat[2*spline+offset])
            self.__A[self.__nth_row, self._segment(spline):self._segment(spline + 1)] = poly
            self.__b[self.__nth_row, :] = self.__waypoints[spline + offset]
            self.__nth_row += 1

    
    def _segment(self, row):
        return row*self.__number_of_coefficients

    def set_minmax_start_and_goal_derivative(self, min_start_derivative=-1, max_end_derivative=-1):
        self._set_min_start_derivative(min_start_derivative)
        self._set_max_end_derivative(max_end_derivative)

    def _set_max_end_derivative(self, max_end_derivative):
        if max_end_derivative < self.__min_start_derivative:
            self.__max_end_derivative = self._degrees
        else:
            self.__max_end_derivative = np.clip(max_end_derivative+1, self.__min_start_derivative, self._degrees)

    def _set_min_start_derivative(self, min_start_derivative):
        if min_start_derivative < 1:
            self.__min_start_derivative = self._derivative
        else:
            self.__min_start_derivative = np.clip(min_start_derivative, 1, self._degrees)
    
    def generate_start_and_goal_constraints(self):
        for derivative in range(self.__min_start_derivative, self.__max_end_derivative):
            polystart = self.generate_polynomial(derivative=derivative, T=0)
            self.__A[self.__nth_row, :self.__number_of_coefficients] = polystart
            self.__nth_row += 1
            polygoal = self.generate_polynomial(derivative=derivative, T=self.__durations[-1])
            self.__A[self.__nth_row, -self.__number_of_coefficients:] = polygoal
            self.__nth_row += 1

    def get_waypoints(self):
        return self.__waypoints

    def get_durations(self):
        return self.__durations

    def get_number_of_waypoints(self):
        return self.__number_of_waypoints

    def get_number_of_splines(self):
        return self.__number_of_splines

    def get_dimensions(self):
        return self.__dimensions

    def get_order(self):
        return self._degrees
    
    def get_degrees(self):
        return self._degrees
    
    def get_number_of_coefficients(self):
        return self.__number_of_coefficients
    
    def get_start_point(self):
        return self.__b[0] if self.__b is not None else self.__waypoints[0]

    def get_end_point(self):
        return self.__b[self.__number_of_waypoints] if self.__b is not None else self.__waypoints[-1]

    def A_matrix(self):
        return self.__A
    
    def b_matrix(self):
        return self.__b

    def get_time_matrix(self):
        return self.__time_matrix
    
    def get_nth_row(self):
        return self.__nth_row

    def get_coefficients(self):
        return self.coefficients
    
if __name__ == "__main__":
    waypoints = np.array([[0, 0, 10], [10, 10, 10], [20, 15, 25], [30, 25, 10], [0, 0, 10]])
    durations = np.array([5, 10, 15, 20])
    mintrajgen = MinimalTrajectoryGenerator(waypoints, durations, Derivative.SNAP)
    mintrajgen.create_poly_matrices()
    mintrajgen.compute_splines()
    trajectory = mintrajgen.get_splines()
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    vx = trajectory[:, 0+3]
    vy = trajectory[:, 1+3]
    vz = trajectory[:, 2+3]
    acx = trajectory[:, 0+6]
    acy = trajectory[:, 1+6]
    acz = trajectory[:, 2+6]

    # plt.rcParams.update({'font.size': 24})
    # fig = plt.figure(figsize=(19.2, 10.8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(waypoints[:, 0], waypoints[:, 1],waypoints[:, 2], label='Waypoint', marker='o', color='crimson')
    # ax.plot(x, y, z, label='Trajectory', marker='o', color='turquoise', alpha=0.2)
    # ax.set_xlabel('X', color="red")
    # ax.set_ylabel('Y', color="green")
    # ax.set_zlabel('Z', color="blue")
    # # Color the axes
    # ax.xaxis.label.set_color('red')
    # ax.yaxis.label.set_color('green')
    # ax.zaxis.label.set_color('blue')

    # # Color the tick labels
    # ax.tick_params(axis='x', colors='red')
    # ax.tick_params(axis='y', colors='green')
    # ax.tick_params(axis='z', colors='blue')
    # ax.set_title('Minimal Snap Trajectory')
    # ax.legend()
    # fig.savefig('minimal_snap_trajectory.png', bbox_inches='tight', dpi=100)
    # plt.show()
    
    time = np.linspace(0, np.sum(durations), len(trajectory))
    plt.rcParams.update({'font.size': 36})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19.2, 10.8))
    # Plot velocities on the top subplot
    ax1.plot(time, vx, label='Velocity X', color='red', linestyle='--', linewidth=5)
    ax1.plot(time, vy, label='Velocity Y', color='green', linestyle='--', linewidth=5)
    ax1.plot(time, vz, label='Velocity Z', color='blue', linestyle='--', linewidth=5)
    ax1.set_ylabel('Velocity [m/s]', color="black")
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    ax1.legend(loc='upper right')

    # Plot accelerations on the bottom subplot
    ax2.plot(time, acx, label='Acceleration X', color='red', linestyle='-', marker='*', alpha=0.6)
    ax2.plot(time, acy, label='Acceleration Y', color='green', linestyle='-', marker='*', alpha=0.6)
    ax2.plot(time, acz, label='Acceleration Z', color='blue', linestyle='-', marker='*', alpha=0.6)
    ax2.set_xlabel('Time [s]', color="black")
    ax2.set_ylabel('Acceleration [m/s²]', color="black")
    ax2.tick_params(axis='x', colors='black')
    ax2.tick_params(axis='y', colors='black')
    ax2.legend(loc='upper right')

    # Set the title for the entire figure
    fig.suptitle('Minimal Snap Trajectory Velocities and Accelerations')

    # Save the figure with cropped white space
    fig.savefig('minimal_snap_trajectory_vel_acc.png', bbox_inches='tight', dpi=100)

    # Show the plot
    plt.show()