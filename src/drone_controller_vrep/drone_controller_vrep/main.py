from header_file import *
from polynomialtrajectory import MinimalTrajectoryGenerator as MinTrajGen
import numpy as np


def main():
    waypoints = np.array([[2.48, -1.08, 1.0], [1.93, -1.45, 1.0], [1.55, -1.63, 1.0], [0.8, -2.0, 1.0], [0.2, -1.55, 1.0],[-0.08, -1.38, 1.0], [-0.6, -1.03, 1.0], [-1.25, -0.58, 1.0], [-2.15, 0.03, 1.0], [-1.63, 0.53, 1.0], [-1.18, 1.03, 1.0], [-0.35, 1.4, 1.0], [0.13, 1.63, 1.0], [0.75, 1.95, 1.0], [1.3, 1.53, 1.0], [1.73, 1.03, 1.0], [1.25, 0.43, 1.0], [0.8, 0.03, 1.0], [0.83, -0.4, 1.0], [0.83, -0.95, 1.0], [1.7, -0.98, 1.0], [2.48, -1.08, 1.0]])
    durations = np.array([0.7, 0.2, 0.8, 0.8, 0.3, 0.7, 1.2, 1.5, 1.2, 1.2, 1.5, 0.5, 0.6, 0.6, 0.6, 1, 0.4, 0.3, 0.4, 0.7, 0.5])
    generator = MinTrajGen(waypoints=waypoints, durations=durations, minimal_trajectory_order=Derivative.JERK)
    generator.set_maximum_velocity(3)
    generator.set_dt(0.05)
    generator.create_poly_matrices()
    generator.compute_splines()
    trajectory = generator.get_splines()
    
    plotter = Plotter()    
    plotter.initialize(trajectory, waypoints, durations)
    plotter.plot_3D(save_plot=False)
    plotter.save_data("Minimal_jerk_trajectory_for_pipelines.csv", trajectory)
    plotter.save_data("Minimal_jerk_trajectory_coefficients_for_pipelines.csv", generator.get_coefficients())
    #trajectory = plotter.read_data("MintrajGen_snap")
    #plotter.plot_3D(save_plot=False)
    #plotter.plot_time_data_at_same_time(save_plot=False)
    #plotter.plot_time_data_individually(save_plot=False)
    
    
    #trajectory = plotter.read_data("straight_trajectory.csv")
    #drone_path = plotter.read_data("drone_path_straight.csv")
    #plotter.initialize(trajectory, waypoints, durations, Derivative_data=Derivative.JERK)
    #plotter.append_title_name("with Straight Drone path PID")
    #plotter.plot_2d_trajectory(label = "Trajectory")
    #plotter.set_trajectory(drone_path)
    #plotter.plot_2d_trajectory(label = "Straight drone path PID")
    #plotter.display_labels_2d(save_plot = False)    
    #drone_force_and_torque = plotter.read_data("drone_force_and_torque_snap.csv")
    
    
if __name__ == "__main__":
    main()