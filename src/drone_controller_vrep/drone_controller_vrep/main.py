from header_file import *
from polynomialtrajectory import MinimalTrajectoryGenerator as MinTrajGen
import numpy as np


def main():
    plotter = Plotter()    
    #generate_trajectory(plotter)
    plot_trajectory(plotter)

def generate_trajectory(plotter: Plotter):
    waypoints = np.array([[2.48, -1.08, 1.0], [1.93, -1.45, 1.0], [1.55, -1.63, 1.0], [0.8, -2.0, 1.0], [0.2, -1.55, 1.0],[-0.08, -1.38, 1.0], [-0.6, -1.03, 1.0], [-1.25, -0.58, 1.0], [-2.15, 0.03, 1.0], [-1.63, 0.53, 1.0], [-1.18, 1.03, 1.0], [-0.35, 1.4, 1.0], [0.13, 1.63, 1.0], [0.75, 1.95, 1.0], [1.3, 1.53, 1.0], [1.73, 1.03, 1.0], [1.25, 0.43, 1.0], [0.8, 0.03, 1.0], [0.83, -0.4, 1.0], [0.83, -0.95, 1.0], [1.7, -0.98, 1.0], [2.48, -1.08, 1.0]])
    durations = np.array([0.7, 0.2, 0.8, 0.8, 0.3, 0.7, 1.2, 1.5, 1.2, 1.2, 1.5, 0.5, 0.6, 0.6, 0.6, 1, 0.4, 0.3, 0.4, 0.7, 0.5])
    generator = MinTrajGen(waypoints=waypoints, durations=durations, minimal_trajectory_order=Derivative.SNAP)
    generator.set_maximum_velocity(3)
    generator.set_dt(0.05)
    generator.create_poly_matrices()
    generator.compute_splines()
    trajectory = generator.get_splines()
    plotter.initialize(trajectory, waypoints, durations)
    plotter.plot_3D(save_plot=False)
    plotter.save_data("Minimal_snap_trajectory_for_pipelines.csv", trajectory)
    plotter.save_data("Minimal_snap_trajectory_coefficients_for_pipelines.csv", generator.get_coefficients())
    

def plot_trajectory(plotter: Plotter):
    trajectory = plotter.read_data("Minimal_snap_trajectory_for_pipelines.csv")
    waypoints = plotter.read_data("Waypoints.csv")
    durations = plotter.read_data("Durations.csv")
    drone_path = plotter.read_data("drone_path_jerk_OIAC.csv")
    drone_force_and_torque = plotter.read_data("drone_force_and_torque_jerk_OIAC.csv")
    drone_k_values = plotter.read_data("drone_K_values.csv")
    drone_d_values = plotter.read_data("drone_D_values.csv")
    
    plotter.initialize(trajectory, waypoints, durations, Derivative_data=Derivative.JERK)
    #plotter.plot_3D(save_plot=False)
    #plotter.plot_time_data_at_same_time(save_plot=False)
    #plotter.plot_time_data_individually(save_plot=False)
    #plotter.append_title_name("with Straight Drone path PID")
    #plotter.plot_2d_trajectory(label = "Trajectory")
    #plotter.set_trajectory(drone_path)
    #plotter.plot_2d_trajectory(label = "Straight drone path PID")
    #plotter.display_labels_2d(save_plot = False)  
    #plot_KD_Terms(plotter, waypoints, durations, drone_d_values, "D")
    find_KD_values(drone_k_values, drone_d_values)
    

def find_KD_values(kval, dval):
    find_KD_minmaxmean("K throttle", kval, "K xy", "K rpy")
    find_KD_minmaxmean("D throttle", dval, "D xy", "D rpy")


def find_KD_minmaxmean(arg0, arg1, arg2, arg3):
    minmaxmean(arg0, arg1[:, 0])
    minmaxmean(arg2, arg1[:, 1:3])
    minmaxmean(arg3, arg1[:, 3:6])

def minmaxmean(name, data):
    print(f"{name} Max: {np.max(data)}")
    print(f"{name} Min: {np.min(data)}")
    print(f"{name} Mean: {np.mean(data)}")

def plot_KD_Terms(plotter, waypoints, durations, data, letter):
    plotter.initialize(data, waypoints, durations)
    plotter.set_title(f"Minimal jerk trajectory with Drone {letter} values for throttle")
    plotter.add_other_data(other_data=data[: , 0], name=f"{letter} values for throttle")
    plotter.plot_other_data_vs_time(length=0)
    plotter.set_title(f"Minimal jerk trajectory with Drone {letter} values for x and y")
    plotter.add_other_data(other_data=data[: , 1], name=f"{letter} values for outer loop x")
    plotter.add_other_data(other_data=data[: , 2], name=f"{letter} values for outer loop y")
    plotter.plot_other_data_vs_time(length=[1, 2])
    plotter.set_title(f"Minimal jerk trajectory with Drone {letter} values for roll, pitch and yaw")
    plotter.add_other_data(other_data=data[: , 3], name=f"{letter} values for inner loop roll")
    plotter.add_other_data(other_data=data[: , 4], name=f"{letter} values for inner loop pitch")
    plotter.add_other_data(other_data=data[: , 5], name=f"{letter} values for inner loop yaw")
    plotter.plot_other_data_vs_time(length=range(3, 6))
    
    
if __name__ == "__main__":
    main()