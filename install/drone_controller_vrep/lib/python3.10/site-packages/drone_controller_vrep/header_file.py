import numpy as np
from numpy.testing import *
import matplotlib.pyplot as plt
import csv
import os
from enum import Enum
class Degree(Enum):
    CONSTANT = 0
    LINEAR = 1
    QUADRATIC = 2
    CUBIC = 3
    QUARTIC = 4
    QUINTIC = 5
    SEXTIC = 6
    SEPTIC = 7
    OCTIC = 8
    NONIC = 9
    DECIC = 10
    UNDECIC = 11
    DUODECIC = 12

class Derivative(Enum):
    POSITION = 0
    VELOCITY = 1
    ACCELERATION = 2
    JERK = 3
    SNAP = 4
    CRACKLE = 5
    POP = 6
    
class Dimension(Enum):
    X = 0
    Y = 1
    Z = 2

class Color(Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
    @classmethod
    def get_color(cls, index):
        members = list(cls)
        return members[index % 3].name.lower()

def ensure_numpy_array(array_or_scalar):
    return np.asarray(array_or_scalar)

def diff2deg(order_type):
    if order_type.__class__.__name__ == 'Derivative':
        return Degree(max(order_type.value * 2 - 1, 0))
    else:
        raise ValueError("Invalid data type")
    

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
        self.initialize_title()
    
    
    def initialize_title(self):
        self.__title = f"Minimal {self.__derivative_type.name.lower()} trajectory"
        
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
        max_derivative_value = self._check_max_derivative_value()
        self._set_max_derivative_and_degrees(max_derivative_value)
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
            self.time_data = np.linspace(0, sum(self.__duration_data), self.__length)
            self.__durations = np.cumsum(np.append(0, self.__duration_data))

    def plot_2D(self, **kwargs):
        self.plot_2D_data(**kwargs)
        self.display_labels_2d(**kwargs)

    def plot_2D_data(self, **kwargs):
        self._pop_multiple_keys(kwargs, ['save_plot', 'show_plot', 'overlap_plot'])
        self.plot_2d_trajectory(**kwargs)
        self.plot_2d_waypoints(**kwargs)

    def display_labels_2d(self, **kwargs):
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        self.append_title_name("2D")
        plt.title(self.__title)
        plt.legend()
        self._save_space_plot(**kwargs)
        plt.show()

    def _save_space_plot(self, **kwargs):
        if kwargs.get('save_plot') is True:
            self.save_image(self.__title)

    def plot_2d_waypoints(self, **kwargs):
        plt.scatter(self.__waypoint_data[:, 0], self.__waypoint_data[:, 1], label='Waypoints', **kwargs)

    def plot_2d_trajectory(self, **kwargs):
        label = kwargs.pop('label', 'Trajectory')
        plt.plot(self.__trajectory_data[:, 0], self.__trajectory_data[:, 1], label=label, **kwargs)

    def plot_3D(self, **kwargs):
        if kwargs.get('overlap_plot') is True or self.__ax_3d is None:
            self.set_3d_figure()
        self.plot_3d_data(**kwargs)
        if kwargs.get('show_plot') is True:
            self.display_labels_3d(**kwargs)

    def set_3d_figure(self):
        self.__ax_3d = plt.figure().add_subplot(111, projection='3d')

    def plot_3d_data(self, **kwargs):
        self._pop_multiple_keys(kwargs, ['save_plot', 'show_plot', 'overlap_plot'])
        self.plot_3d_trajectory(label='Trajectory', **kwargs)
        self.plot_3d_waypoints(label='Waypoints', **kwargs)

    def display_labels_3d(self, **kwargs):
        self.show_labels_3D_plots()
        self.save_3D_plot(kwargs)
        plt.show()

    def save_3D_plot(self, kwargs):
        if kwargs.get('save_plot') is True:
            self.save_image(self.__title)

    def show_labels_3D_plots(self) :
        self.__ax_3d.set_xlabel("X-axis [m]")
        self.__ax_3d.set_ylabel("Y-axis [m]")
        self.__ax_3d.set_zlabel("Z-axis [m]")
        self.append_title_name("3D")
        self.__ax_3d.set_title(f"{self.__title}")
        self.__ax_3d.legend()

    def save_image(self, name):
        plt.savefig(self.get_file_location(f"{name}.png", 'resource/img'))
        
    def plot_3d_waypoints(self, **kwargs):
        self.__ax_3d.scatter(self.__waypoint_data[:, 0], self.__waypoint_data[:, 1], self.__waypoint_data[:, 2], **kwargs)

    def plot_3d_trajectory(self, **kwargs):
        self.__ax_3d.plot(self.__trajectory_data[:, 0], self.__trajectory_data[:, 1], self.__trajectory_data[:, 2], **kwargs)


    def read_data(self, filename, order=-1, dimensions=-1):
        filename = self.get_file_location(filename, 'resource/data')
        with open(filename, 'r') as file:
            return self.read_file(file, order, dimensions)

    def read_file(self, file, order=-1, dimensions=-1):
        reader = csv.reader(file)
        data_list = list(reader)
        data_array = np.array(data_list).astype(np.float64)
        return data_array[:, order:order+dimensions] if order > -1 and dimensions > -1 else data_array

    def get_file_location(self, filename, location='data'):
        directory_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), location)
        os.makedirs(directory_path, exist_ok=True)
        return os.path.join(directory_path, filename)

    def save_data(self, filename, DATA):
        np.savetxt(self.get_file_location(filename), DATA, delimiter=",", fmt='%f')

    def plot_time_data_individually(self, **kwargs):
        for derivative in range(self.__max_derivative.value + 1):
            axs = self.get_subplot_for_individual_plots()
            self._plot_time_vs_trajectory_and_waypoint(axs, derivative, **kwargs)
            plt.tight_layout()
            self._save_time_plot(derivative, "individual", **kwargs)
        plt.show()

    def _save_time_plot(self, derivative, keyname, **kwargs):
        if kwargs.get('save_plot') is True:
            self.save_image(f"Minimal {self.__derivative_type.name.lower()} trajectory {Derivative(derivative).name.lower()} {keyname}")

    def get_subplot_for_individual_plots(self):
        _, axs = plt.subplots(self.__dimensions, 1, figsize=(10, self.__dimensions * 4))
        return np.array(axs).reshape(-1)

    def _plot_time_vs_trajectory_and_waypoint(self, axs, derivative, **kwargs):
        self._pop_multiple_keys(kwargs, ['save_plot'])
        for dimension in range(self.__dimensions):
            self._plot_time_vs_trajectory(dimension + derivative*self.__dimensions, axs[dimension],  label=f"{Dimension(dimension).name} {Derivative(derivative).name.lower()}", color=Color.get_color(dimension), **kwargs)
            if derivative == 0:
                self._plot_durations_vs_waypoints(dimension, axs[dimension], color=Color.get_color(dimension), **kwargs)
            self.set_label_for_time_plot(dimension, derivative, axs)
            

    def set_label_for_time_plot(self, i, derivative,  axs):
        axs[i].set_title(f"Time vs {Dimension(i).name} {Derivative(derivative).name.lower()}")
        axs[i].set_ylabel(self._format_unit_label(f'{Derivative(derivative).name.lower().capitalize()}', 0))
        axs[i].set_xlabel("Time [s]")
        axs[i].legend()

    def _plot_time_vs_trajectory(self, dimension, ax, **kwargs):
        ax.plot(self.time_data, self.__trajectory_data[:, dimension], **kwargs)

    def _plot_durations_vs_waypoints(self, dimension,  ax, **kwargs):
        ax.scatter(self.__durations, self.__waypoint_data[:, dimension], **kwargs)


    def _pop_multiple_keys(self, dictionary, keys, default=None):
        return [dictionary.pop(key, default) for key in keys]

    def plot_time_data_at_same_time(self, **kwargs):
        for derivative in range(self.__max_derivative.value + 1):
            _, ax = plt.subplots(figsize=(10, 6))
            self._plot_time_vs_data_at_same_time(derivative, ax, **kwargs)
            self._set_label_for_time_vs_data_at_same_time(derivative, ax)
            plt.tight_layout()
            self._save_time_plot(derivative, "simultaneous", **kwargs)
        plt.show()

    def _set_label_for_time_vs_data_at_same_time(self, derivative, ax):
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(self._format_unit_label(f'{Derivative(derivative).name.lower().capitalize()}', 0))
        ax.set_title(f"{self.__title}: Time vs {Derivative(derivative).name.lower()}")
        ax.legend()


    def _plot_time_vs_data_at_same_time(self, derivative, ax, **kwargs):
        self._pop_multiple_keys(kwargs, ['save_plot'])
        for dimension in range(self.__dimensions):
            self._plot_time_vs_trajectory(dimension + derivative*self.__dimensions, ax, label=f'{Dimension(dimension).name} trajectory', color=Color.get_color(dimension), **kwargs)
            if derivative == 0:
                self._plot_durations_vs_waypoints(dimension, ax, label=f'{Dimension(dimension).name} waypoint', color=Color.get_color(dimension), **kwargs)
            
            

    def _format_unit_label(self, derivative_name, derivative_order):
        unit_latex_str = self._format_meters_per_seconds(derivative_order)
        return f"{derivative_name} $\\left[{unit_latex_str}\\right]$"

    def _format_meters_per_seconds(self, derivative_order):
        return {
            0: "m",
            1: f"\\frac{{m}}{{s}}",
        }.get(derivative_order, f"\\frac{{m}}{{s^{{{derivative_order}}}}}")

