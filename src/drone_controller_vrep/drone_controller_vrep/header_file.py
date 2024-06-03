import numpy as np
from numpy.testing import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
import matplotlib.transforms as mtransforms
import csv
import os
from enum import Enum
import pandas as pd
import itertools
from itertools import cycle
import random

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

class Controller(Enum):
    OIAC = 0
    PID = 2
    MRAC = 1



@staticmethod
def ensure_numpy_array(array_or_scalar):
    return np.asarray(array_or_scalar)

@staticmethod
def diff2deg(order_type):
    if order_type.__class__.__name__ == 'Derivative':
        return Degree(max(order_type.value * 2 - 1, 0))
    else:
        raise ValueError("Invalid data type")

@staticmethod
def get_file_location(filename, location='data'):
    directory_path = os.path.join('src/drone_controller_vrep', location)
    os.makedirs(directory_path, exist_ok=True)
    return os.path.join(directory_path, filename)

@staticmethod
def read_data(filename, order=-1, dimensions=-1):
    filename = get_file_location(filename, 'resource/data')
    with open(filename, 'r') as file:
        return read_file(file, order, dimensions)

@staticmethod
def read_file(file, order=-1, dimensions=-1):
    reader = csv.reader(file)
    data_list = list(reader)
    data_array = np.array(data_list).astype(np.float64)
    return data_array[:, order:order+dimensions] if order > -1 and dimensions > -1 else data_array

@staticmethod
def save_data(filename, DATA):
    np.savetxt(get_file_location(filename, location='resource/data'), DATA, delimiter=",", fmt='%f')


@staticmethod
def save_image(name):
    plt.savefig(get_file_location(f"{name}.png", 'resource/img'))

@staticmethod
def extract_rrt_star_array(*best_path):
    if len(best_path) == 1:
        temp_array = best_path[0]
    if isinstance(temp_array, list):
        temp_array = prune_array(np.concatenate(temp_array))
    return temp_array

@staticmethod
def prune_array(data, tolerance=1e-5):
    if isinstance(data, list):
        return np.array(_prune_list(data, tolerance))
    return data if data.size == 0 else np.array(_prune_data(data, tolerance))

@staticmethod
def _prune_list(data, tolerance=1e-5):
    pruned_data = []
    for datum in data:
        for point in datum:
            if (
                not pruned_data
                or np.linalg.norm(pruned_data[-1] - point) > tolerance
            ):
                pruned_data.append(point)
    return pruned_data
    
@staticmethod
def _prune_data(data, tolerance):
    pruned_data = [data[0]]
    for point in data[1:]:
        if np.linalg.norm(pruned_data[-1] - point) > tolerance:
            pruned_data.append(point)
    return pruned_data

@staticmethod
def calculate_distance_error(trajectory, drone_path):
    trajectory, drone_path = equalize_data_length(trajectory[:, :3], drone_path[:, :3])
    return np.linalg.norm(trajectory - drone_path, axis=1)

@staticmethod
def equalize_data_length(data1, data2):
        min_length = min(len(data1), len(data2))
        data1 = data1[-min_length:, :]
        data2 = data2[-min_length:, :]
        return data1, data2
    
    
@staticmethod
def find_and_print_KD_values(kval, dval, derivative, gaussian=0.0):
    find_KD_minmaxmean("K throttle", kval, "K xy", "K rpy", derivative, gaussian)
    find_KD_minmaxmean("D throttle", dval, "D xy", "D rpy", derivative, gaussian)

@staticmethod
def find_KD_minmaxmean(arg0, arg1, arg2, arg3, derivative, gaussian=0.0):
    minmaxmean(arg0, arg1[:, 0], derivative, gaussian)
    minmaxmean(arg2, arg1[:, 1:3], derivative, gaussian)
    minmaxmean(arg3, arg1[:, 3:6], derivative, gaussian)

@staticmethod
def minmaxmean(name, data, derivative='POSITION', gaussian=0.0):
    append = f" with std {gaussian}" if gaussian != 0.0 else ""
    min_data = np.min(data)
    max_data = np.max(data)
    avg_data = np.mean(data)
    with open(get_file_location(f'Drone {name} values {derivative}{append}.txt', 'resource/data'), 'w') as file:
        file.write(f"{derivative}{append} {name} Max: {max_data}\n")
        file.write(f"{derivative}{append} {name} Min: {min_data}\n")
        file.write(f"{derivative}{append} {name} Avg: {avg_data}\n")
    print(f"{derivative}{append} {name} Max: {max_data}")
    print(f"{derivative}{append} {name} Min: {min_data}")
    print(f"{derivative}{append} {name} Avg: {avg_data}")


@staticmethod
def save_optimal_trajectory(trajectory: np.ndarray, order: Derivative):
    save_data(f"rrt_trajectory_{order.name}.csv", trajectory)