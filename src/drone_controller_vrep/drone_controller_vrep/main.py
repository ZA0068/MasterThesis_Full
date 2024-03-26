from header_file import *
from polynomialtrajectory import MinimalTrajectoryGenerator as MinTrajGen
import numpy as np
from rrtstar import RRTStar
from rrtstarplotter import RRTPlotter
from mayavi import mlab

def main():
    #waypoints = np.array([[2.48, -1.08, 1.0], [1.93, -1.45, 1.0], [1.55, -1.63, 1.0], [0.8, -2.0, 1.0], [0.2, -1.55, 1.0],[-0.08, -1.38, 1.0], [-0.6, -1.03, 1.0], [-1.25, -0.58, 1.0], [-2.15, 0.03, 1.0], [-1.63, 0.53, 1.0], [-1.18, 1.03, 1.0], [-0.35, 1.4, 1.0], [0.13, 1.63, 1.0], [0.75, 1.95, 1.0], [1.3, 1.53, 1.0], [1.73, 1.03, 1.0], [1.25, 0.43, 1.0], [0.8, 0.03, 1.0], [0.83, -0.4, 1.0], [0.83, -0.95, 1.0], [1.7, -0.98, 1.0], [2.48, -1.08, 1.0]])
    #rrt_plottingn(waypoints)
    plotter = Plotter()    
    #generate_trajectory(plotter)
    plot_trajectory(plotter)

def rrt_plottingn(waypoints):
    ceiling = [-10, 10, -10, 10, 9, 10]
    floor = [-10, 10, -10, 10, -10, -9]
    space_limits = np.array([[-3., -3., -3], [3., 3., 3.]])
    #obstacles = np.array(
#        [floor,
#        ceiling,
#        [4, 6, 3, 5, 0, 5],
#        [5, 8, 2, 5, 0, 5],
#        [1, 3, 3, 5, 0, 5],
#        [4, 8, 7, 9, 0, 5],
#        ]
#    )
    for i, waypoint in enumerate(waypoints):
        if i == len(waypoints) - 1:
            break
        rrt = RRTStar(
            space_limits,
            start=waypoint,
            goal=waypoints[i+1],
            max_distance=0.8,
            max_iterations=1000,
            obstacles=None,
        )
        rrt.run()
    
    rrt_plotter = RRTPlotter(rrt, None, None)
    #rrt_plotter.plot_obstacles(obstacles)
    rrt_plotter.plot_start_and_goal()
    rrt_plotter.plot_path()
    rrt_plotter.plot_tree()
    
    mlab.orientation_axes()
    mlab.axes()
    mlab.show()


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
    plotter.plot_3D(save_plot=True)
    plotter.plot_time_data_at_same_time(save_plot=True)
    plotter.save_data("Minimal_snap_trajectory_for_pipelines.csv", trajectory)
    plotter.save_data("Minimal_snap_trajectory_coefficients_for_pipelines.csv", generator.get_coefficients())


def plot_trajectory(plotter: Plotter):
    trajectory = plotter.read_data("Minimal_jerk_trajectory_for_pipelines.csv")
    waypoints = plotter.read_data("Waypoints.csv")
    durations = plotter.read_data("Durations.csv")
    drone_path = plotter.read_data("drone_path_jerk_OIAC.csv")
    drone_force_and_torque = plotter.read_data("drone_force_and_torque_snap_OIAC.csv")
    drone_k_values = plotter.read_data("drone_K_values_snap.csv")
    drone_d_values = plotter.read_data("drone_D_values_snap.csv")

    #plot_KD_Terms(plotter, waypoints, durations, drone_k_values, "K")
    #plot_KD_Terms(plotter, waypoints, durations, drone_d_values, "D")
    #find_KD_values(drone_k_values, drone_d_values)
    #plot2D_drone_path(plotter, trajectory, waypoints, durations, drone_path, Derivative.JERK, "OIAC")
    #plot_drone_force_and_torque(plotter, waypoints, durations, drone_force_and_torque, Derivative.JERK, "OIAC")
    plot_2D_distance_error(plotter, trajectory, drone_path, waypoints, durations, Derivative.JERK, "OIAC")

def plot2D_drone_path(plotter, trajectory, waypoints, durations, drone_path, derivative_data, controller):
    plotter.initialize(trajectory, waypoints, durations, derivative_data)
    plotter.append_title_name(f"with drone path {controller}")
    plotter.plot_2d_trajectory(label = "Trajectory")
    plotter.plot_2d_waypoints()
    plotter.set_trajectory(drone_path)
    plotter.plot_2d_trajectory(label = f"drone path {controller}")
    plotter.display_labels_2d(save_plot = True)

def plot_drone_force_and_torque(plotter, waypoints, durations, drone_force_and_torque, derivative_data, controller):
    plotter.initialize(drone_force_and_torque, waypoints, durations, derivative_data)
    plotter.add_other_data(drone_force_and_torque[:, 3],f"{controller} Minimal {str(derivative_data.name).lower()} throttle")
    plotter.set_title(f"Minimal {str(derivative_data.name).lower()} throttle {controller}")
    plotter.plot_other_data_vs_time(index=0)
    plotter.add_other_data(drone_force_and_torque[:, 0],f"{controller} Minimal {str(derivative_data.name).lower()} Roll")
    plotter.add_other_data(drone_force_and_torque[:, 1],f"{controller} Minimal {str(derivative_data.name).lower()} Pitch")
    plotter.add_other_data(drone_force_and_torque[:, 2],f"{controller} Minimal {str(derivative_data.name).lower()} Yaw")
    plotter.set_title(f"Minimal {str(derivative_data.name).lower()} moments {controller}")
    plotter.plot_other_data_vs_time(index=range(1, 4))

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
    print(f"{name} Avg: {np.mean(data)}")

def plot_KD_Terms(plotter: Plotter, waypoints, durations, data, letter):
    plotter.initialize(data, waypoints, durations)
    plotter.set_title(f"Minimal jerk trajectory with Drone {letter} values for throttle")
    plotter.add_other_data(other_data=data[: , 0], name=f"{letter} values for throttle")
    plotter.plot_other_data_vs_time(index=0)
    plotter.set_title(f"Minimal jerk trajectory with Drone {letter} values for x and y")
    plotter.add_other_data(other_data=data[: , 1], name=f"{letter} values for outer loop x")
    plotter.add_other_data(other_data=data[: , 2], name=f"{letter} values for outer loop y")
    plotter.plot_other_data_vs_time(index=[1, 2])
    plotter.set_title(f"Minimal jerk trajectory with Drone {letter} values for roll, pitch and yaw")
    plotter.add_other_data(other_data=data[: , 3], name=f"{letter} values for inner loop roll")
    plotter.add_other_data(other_data=data[: , 4], name=f"{letter} values for inner loop pitch")
    plotter.add_other_data(other_data=data[: , 5], name=f"{letter} values for inner loop yaw")
    plotter.plot_other_data_vs_time(index=range(3, 6))
    
def plot_2D_distance_error(plotter: Plotter, trajectory, drone_path, waypoints, durations, derivative: Derivative, controller):
    plotter.initialize(trajectory, waypoints, durations, derivative)
    plotter.set_title(f"Minimal {derivative.name.lower()} distance error for drone path {controller}")
    plotter.plot_2D_distance_error(trajectory, drone_path, save_plot=True)
    
if __name__ == "__main__":
    main()