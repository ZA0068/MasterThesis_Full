from .header_file import *

class Plotter:
    def __init__(self, Trajectory_data=None, Waypoint_data=None, Duration_data=None, Derivative_data=None) -> None:
        self.reset()
        self.initialize(Trajectory_data, Waypoint_data, Duration_data, Derivative_data)

    def initialize(self, Trajectory_data, Waypoint_data, Duration_data, Derivative_data=None):
        self.set_data(Trajectory_data, Waypoint_data, Duration_data)
        self._extract_data_size()
        self._extract_dimensions()
        self._extract_derivatives_and_degrees(Derivative_data)
        self._set_time_data()
        self._initialize_title()
    
    
    def _initialize_title(self):
        self.set_title(f"Minimal {self.__derivative_type.name.lower()} trajectory")
        
    def set_title(self, title):
        self.__title = title
        
    def reset(self):
        self.__trajectory_data = None
        self.__waypoint_data = None
        self.__duration_data = None
        self.__dimensions = 0
        self.__length = 0
        self.__width = 0
        self.__max_derivative = Derivative(0)
        self.__max_degree = Degree(0)
        self.__derivative_type = Derivative(0)
        self.__degree_type = Degree(0)
        self.__title = ""
        self.__ax_3d = None
        self.__ax_2d = None
        self.__other_data = []
        self.__other_data_counter = 0
    
    def add_other_data(self, other_data, name = ""):
        self.__other_data.append([name, other_data])
        self.__other_data_counter += 1
        
    def _extract_dimensions(self):
        if self.__waypoint_data is None:
            self.__dimensions = 0
        else:
            self.__dimensions = self.__waypoint_data.shape[1]

    def _extract_data_size(self):
        if self.__trajectory_data is not None:
            self.__length = self.__trajectory_data.shape[0]
            self.__width = self.__trajectory_data.shape[1]
        else:
            self.__length = 0
            self.__width = 0

    def set_trajectory(self, trajectory_data):
        self.__trajectory_data = trajectory_data
        
    def set_waypoints(self, waypoint_data):
        self.__waypoint_data = waypoint_data
        
    def set_durations(self, duration_data):
        self.__duration_data = duration_data

    def _extract_derivatives_and_degrees(self, derivative_data):
        self._set_max_derivative_and_degrees(self._check_max_derivative_value())
        self._set_degrees_and_derivative_type(derivative_data)

    def _set_degrees_and_derivative_type(self, derivative_data):
        self.__derivative_type = derivative_data if derivative_data is not None else self.__max_derivative
        self.__degree_type = diff2deg(derivative_data) if derivative_data is not None else self.__max_degree

    def _set_max_derivative_and_degrees(self, max_derivative_value):
        self.__max_derivative = Derivative(max_derivative_value)
        self.__max_degree = Degree(max(self.__max_derivative.value * 2 - 1, 0))

    def _check_max_derivative_value(self):
        return (
            0
            if self.__width == 0 or self.__dimensions == 0
            else round(self.__width / self.__dimensions / 2)
        )

    def append_title_name(self, append=""):
        self.__title = f"{self.__title} {append}"

    def set_data(self, trajectory_data, waypoint_data, duration_data):
        self.set_trajectory(trajectory_data)
        self.set_waypoints(waypoint_data)
        self.set_durations(duration_data)

    def _set_time_data(self):
        if self.__duration_data is not None:
            self.__time_data = np.linspace(0, sum(self.__duration_data), self.__length)
            self.__durations = np.cumsum(np.append(0, self.__duration_data))

    def plot_2D(self, **kwargs):
        self.plot_2D_data(**kwargs)
        self.display_labels_2d(**kwargs)

    def plot_2D_data(self, **kwargs):
        trajectory_label, waypoint_label =self._pop_multiple_keys(kwargs, ['trajectory_label', 'waypoint_label'], ['Optimal Trajectory', 'Waypoints'])
        plt.figure(figsize=(10*2, 8*2))
        plt.rcParams.update({'font.size': 24})
        self.plot_2d_trajectory(label=trajectory_label)
        self.plot_2d_waypoints(label=waypoint_label)

    def display_labels_2d(self, **kwargs):
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        self.append_title_name("2D")
        plt.title(self.__title)
        plt.legend()
        self.save_plot(**kwargs)
        plt.show()

    def save_plot(self, **kwargs):
        if kwargs.get('save_plot') is True:
            self.save_image(self.__title)

    def plot_2d_waypoints(self, **kwargs):
        plt.scatter(self.__waypoint_data[:, 0], self.__waypoint_data[:, 1], **kwargs)

    def plot_2d_trajectory(self, **kwargs):
        plt.plot(self.__trajectory_data[:, 0], self.__trajectory_data[:, 1], **kwargs)

    def plot_3D(self, **kwargs):
        self.set_3d_figure()
        self.plot_3d_data(**kwargs)
        self.display_labels_3d(**kwargs)

    def set_3d_figure(self):
        plt.rcParams.update({'font.size': 24})
        self.__ax_3d = plt.figure(figsize=(10*2, 8*2)).add_subplot(111, projection='3d')
        self.__ax_3d.set_xlim([-3, 3])
        self.__ax_3d.set_ylim([-3, 3])
        self.__ax_3d.set_zlim([0, 3])

    def plot_3d_data(self, **kwargs):
        trajectory_label, waypoint_label = self._pop_multiple_keys(kwargs, ['trajectory_label', 'waypoint_label'], ['Trajectory', 'Waypoints'])
        plt.rcParams.update({'font.size': 24})
        self.plot_3d_trajectory(label=trajectory_label)
        self.plot_3d_waypoints(label=waypoint_label)

    def display_labels_3d(self, **kwargs):
        self.show_labels_3D_plots()
        self.save_plot(**kwargs)
        plt.show()

    def show_labels_3D_plots(self) :
        plt.rcParams.update({'font.size': 24})
        self.__ax_3d.set_xlabel("X-axis [m]")
        self.__ax_3d.set_ylabel("Y-axis [m]")
        self.__ax_3d.set_zlabel("Z-axis [m]")
        self.append_title_name("3D")
        self.__ax_3d.set_title(f"{self.__title}")
        self.__ax_3d.legend()
        

    def save_image(self, name):
        plt.savefig(get_file_location(f"{name}.png", 'resource/img'))
        
    def plot_3d_waypoints(self, **kwargs):
        self.__ax_3d.scatter(self.__waypoint_data[:, 0], self.__waypoint_data[:, 1], self.__waypoint_data[:, 2], **kwargs)

    def plot_3d_trajectory(self, **kwargs):
        self.__ax_3d.plot(self.__trajectory_data[:, 0], self.__trajectory_data[:, 1], self.__trajectory_data[:, 2], **kwargs)

    def plot_time_data_individually(self, **kwargs):
        for derivative in range(self.__max_derivative.value + 1):
            axs = self.get_subplot_for_individual_plots()
            self.plot_time_vs_trajectory_and_waypoint(axs, derivative)
            plt.tight_layout()
            self.set_title(f"Minimal {self.__derivative_type.name.lower()} trajectory {Derivative(derivative).name.lower()} individual")
            plt.rcParams.update({'font.size': 24})
            self.save_plot(**kwargs)
        plt.show()



    def get_subplot_for_individual_plots(self):
        _, axs = plt.subplots(self.__dimensions, 1, figsize=(10, self.__dimensions * 4))
        return np.array(axs).reshape(-1)

    def plot_time_vs_trajectory_and_waypoint(self, axs, derivative):
        for dimension in range(self.__dimensions):
            self.plot_time_vs_trajectory(dimension + derivative*self.__dimensions, axs[dimension],  label=f"{Dimension(dimension).name} {Derivative(derivative).name.lower()}", color=Color.get_color(dimension))
            if derivative == 0:
                self.plot_durations_vs_waypoints(dimension, axs[dimension], color=Color.get_color(dimension))
            self.set_label_for_time_plot(dimension, derivative, axs)
            

    def set_label_for_time_plot(self, i, derivative,  axs):
        axs[i].set_title(f"Time vs {Dimension(i).name} {Derivative(derivative).name.lower()}")
        axs[i].set_ylabel(self._format_unit_label(f'{Derivative(derivative).name.lower().capitalize()}', 0))
        axs[i].set_xlabel("Time [s]")
        axs[i].legend()

    def plot_time_vs_trajectory(self, dimension, ax, **kwargs):
        ax.plot(self.__time_data, self.__trajectory_data[:, dimension], **kwargs)

    def plot_durations_vs_waypoints(self, dimension,  ax, **kwargs):
        ax.scatter(self.__durations, self.__waypoint_data[:, dimension], **kwargs)


    def _pop_multiple_keys(self, dictionary, keys, defaults=None):
        if defaults is None:
            defaults = [None] * len(keys)
        elif not isinstance(defaults, (list, tuple)):
            defaults = [defaults] * len(keys)
        if len(defaults) != len(keys):
            raise ValueError("Length of defaults must match length of keys")
        return tuple(dictionary.pop(key, default) for key, default in zip(keys, defaults))





    def plot_time_data_at_same_time(self, **kwargs):
        for derivative in range(self.__max_derivative.value + 1):
            _, ax = plt.subplots(figsize=(10*2, 8*2))
            self.plot_time_vs_trajectory_at_same_time(derivative, ax)
            self.set_label_for_time_vs_trajectory_at_same_time(derivative, ax)
            plt.tight_layout()
            self.set_title(f"Minimal {self.__derivative_type.name.lower()} trajectory {Derivative(derivative).name.lower()} simultaneous")
            plt.rcParams.update({'font.size': 24})
            self.save_plot(**kwargs)
        plt.show()

    def set_label_for_time_vs_trajectory_at_same_time(self, derivative, ax):
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(self._format_unit_label(f'{Derivative(derivative).name.lower().capitalize()}', 0))
        ax.set_title(f"{self.__title}: Time vs {Derivative(derivative).name.lower()}")
        ax.legend()


    def plot_time_vs_trajectory_at_same_time(self, derivative, ax):
        for dimension in range(self.__dimensions):
            self.plot_time_vs_trajectory(dimension + derivative*self.__dimensions, ax, label=f'{Dimension(dimension).name} trajectory', color=Color.get_color(dimension))
            if derivative == 0:
                self.plot_durations_vs_waypoints(dimension, ax, label=f'{Dimension(dimension).name} waypoint', color=Color.get_color(dimension))
            
    def plot_other_data_vs_time(self, **kwargs):
        plt.rcParams.update({'font.size': 24})
        _, ax = plt.subplots(figsize=(10*2, 8*2))
        log_scale = kwargs.pop('log_scale',False)
        length = np.array(kwargs.pop('index', range(self.__other_data_counter))).flatten()
        for cnt in length:
            ax.plot(self.__time_data[:len(self.__other_data[cnt][1])], self.__other_data[cnt][1], label=self.__other_data[cnt][0])
        ax.set_xlabel("Time [s]")
        ylabel = kwargs.pop('ylabel', 'Value')
        ax.set_ylabel(ylabel)
        ax.set_title(self.__title)
        if log_scale:
            ax.set_yscale('log')
        ax.legend()
        self.save_plot(**kwargs)
        plt.show()
        


    def plot_2D_distance_error(self, trajectory, drone_path,  **kwargs):
        plt.rcParams.update({'font.size': 24})
        _, ax = plt.subplots(figsize=(10*2, 8*2))
        self.plot_distance_error(ax, trajectory, drone_path)
        self.set_label_for_distance_error(ax)
        self.save_plot(**kwargs)
        plt.show()

    def plot_1d_distance_error(self, trajectory, drone_path, dimension, **kwargs):
        _, ax = plt.subplots(figsize=(10*2, 8*2))
        self.plot_distance_error(ax, trajectory[:, dimension], drone_path[:, dimension], **kwargs)
        self.set_label_for_distance_error(ax)
        self.save_plot(**kwargs)
        plt.show()
            
    def set_label_for_distance_error(self, ax):
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error Distance [m]")
        ax.set_title(self.__title)
        ax.legend()
        
    def plot_distance_error(self, ax, trajectory, drone_path):
        distance_error = calculate_distance_error(trajectory, drone_path)
        ax.plot(self.__time_data[:len(distance_error)], distance_error, label='Distance error')
        ax.text(20.24, 0.4, f'AVG error: {np.mean(distance_error):.2f}', fontsize=24, bbox=dict(facecolor='red', alpha=0.5))
        ax.text(20.24, 0.3, f'STD error: {np.std(distance_error):.2f}', fontsize=24, bbox=dict(facecolor='blue', alpha=0.5))
        

    

    def _format_unit_label(self, derivative_name, derivative_order):
        unit_latex_str = self._format_meters_per_seconds(derivative_order)
        return f"{derivative_name} $\\left[{unit_latex_str}\\right]$"

    def _format_meters_per_seconds(self, derivative_order):
        return {
            0: "m",
            1: f"\\frac{{m}}{{s}}",
        }.get(derivative_order, f"\\frac{{m}}{{s^{{{derivative_order}}}}}")