from header_file import *
from polynomialtrajectory import MinimalTrajectoryGenerator as MinTrajGen
import numpy as np
from rrtstar import RRTStar
from rrtstarplotter import RRTPlotter
from obstacle import Obstacle

class Runner():
    def __init__(self):
        self.__plotter = Plotter()
        self.__rrt = RRTStar()
        self.__rrtplotter = RRTPlotter()
        self.__mintrajgen = MinTrajGen()
        self.__true_rrt_tree = []
        self.__rrt_waypoints = None
        self.__optimal_trajectory = None
        self.__drone_path_oiac = None
        self.__drone_path_pid = None
        self.__main_waypoints = None
        self.__durations = None
        self.__derivative = None
        self.__controller = None
        
    def set_main_waypoints(self, main_waypoints: np.ndarray):
        self.__main_waypoints = main_waypoints
        
    def set_durations(self, durations: np.ndarray):
        if durations is not None:
            self.__durations = durations
        else:
            self.__durations = read_data("Durations.csv")
    
    def set_rrt(self, rrt: RRTStar):
        self.__rrt = rrt
    
    def set_rrt_waypoints(self, rrt_waypoints: np.ndarray):
        self.__rrt_waypoints = rrt_waypoints
    
    def set_optimal_trajectory(self, optimal_trajectory: np.ndarray):
        self.__optimal_trajectory = optimal_trajectory
        
    def set_drone_path(self, drone_path: np.ndarray):
        self.__drone_path_oiac = drone_path
    
    def set_derivative(self, derivative: Derivative):
        self.__derivative = derivative
    
    def set_controller(self, controller: Controller):
        self.__controller = controller

    def build_rrt(self, run=False):
        self._setup_rrt()
        self._generate_rrt_waypoints(run)

    def _setup_rrt(self):
        self.__rrt.add_obstacles(self._setup_obstacles())
        self.__rrt.set_max_step(0.1)
        self.__rrt.set_boundaries([-3, -3, 0, 3, 3, 5])

    def _setup_obstacles(self):
        obstacle1 = Obstacle([-1.3, -1.4, 0.0, -0.3, -0.4, 1.0])
        obstacle2 = Obstacle([-0.65, 1.05, 0.0, 0.35, 2.05, 1.0]).rotate(25, 0, 0)
        floor = Obstacle([-2.5, -2.5, -0.1, 2.5, 2.5, 0.0])
        return obstacle1,obstacle2,floor

    def _generate_rrt_waypoints(self, run):
        if run:
            self.__rrt_waypoints = prune_array(self._build_rrt())
            save_data("rrt_path.csv", self.__rrt_waypoints)
        else: 
            self.__rrt_waypoints = read_data("rrt_path.csv")

    def _build_rrt(self):
        _rrt_waypoints_local = []
        for i in range(len(self.__main_waypoints) - 1):
            self.__rrt.set_start_and_goal(self.__main_waypoints[i], self.__main_waypoints[i + 1])
            self.__rrt.run()
            _rrt_waypoints_local.append(self.__rrt.get_best_path())
            self.__true_rrt_tree.append(self.__rrt.get_best_tree())
            self.__rrt.reset()
        return _rrt_waypoints_local

    def plot_trajectory_and_drone_data(self, derivative: Derivative, save_plots: bool = False):
        self._set_plot_types(derivative, save_plots)
        self.acquire_trajectory_and_drone_data()
        self.plot_3D_drone_path()
        self.plot_KD_values(self.get_drone_kd_values())
        self.plot_2D_drone_path()
        self.plot_drone_force_and_torque()
        self.plot_distance_error()
        self.plot_rrt()

    def _set_plot_types(self, derivative, save_plots):
        self.__derivative = derivative
        self.__save_plots = save_plots

    
    def acquire_trajectory_and_drone_data(self):
        self.__optimal_trajectory = read_data(f"rrt_trajectory_{self.__derivative.name}.csv")
        self.__drone_path_oiac = read_data(f"drone_path_{self.__derivative.name}_OIAC.csv")
        self.__drone_path_pid = read_data(f"drone_path_{self.__derivative.name}_PID.csv")
        self.__drone_force_and_torque_oiac = read_data(f"drone_force_and_torque_{self.__derivative.name}_OIAC.csv")
        self.__drone_force_and_torque_pid = read_data(f"drone_force_and_torque_{self.__derivative.name}_PID.csv")
        self.__durations = np.ones(self.__drone_path_oiac.shape[0]) * 0.05

    def plot_3D_drone_path(self):
        self.__plotter.initialize(self.__optimal_trajectory, self.__rrt_waypoints, self.__durations, self.__derivative)
        self.__plotter.append_title_name("with rrt-star and drone path")
        self.__plotter.set_3d_figure()
        self.__plotter.plot_3d_data(waypoint_label="rrt-star generated waypoints")
        self.__plotter.set_trajectory(self.__drone_path_oiac)
        self.__plotter.plot_3d_trajectory(label = "drone path OIAC")
        self.__plotter.set_trajectory(self.__drone_path_pid)
        self.__plotter.plot_3d_trajectory(label = "drone path PID")
        self.__plotter.set_waypoints(self.__main_waypoints)
        self.__plotter.plot_3d_waypoints(label='Waypoints')
        self.__plotter.display_labels_3d(save_plot = self.__save_plots)
        self.__plotter.reset()

    def plot_KD_values(self, drone_kd_values):
            find_and_print_KD_values(drone_kd_values[0], drone_kd_values[1])
            self.plot_KD_Terms(drone_kd_values[0], "K")
            self.plot_KD_Terms(drone_kd_values[1], "D")
            
    def plot_KD_Terms(self, data, letter):
        self.__plotter.initialize(data, self.__rrt_waypoints, self.__durations)
        self.__plotter.set_title(f"Minimal {self.__derivative.name.lower()} trajectory with Drone {letter} values for throttle")
        self.__plotter.add_other_data(other_data=data[: , 0], name=f"{letter} values for throttle")
        self.__plotter.plot_other_data_vs_time(index=0, save_plot=True)
        self.__plotter.set_title(f"Minimal {self.__derivative.name.lower()} trajectory with Drone {letter} values for x and y")
        self.__plotter.add_other_data(other_data=data[: , 1], name=f"{letter} values for outer loop x")
        self.__plotter.add_other_data(other_data=data[: , 2], name=f"{letter} values for outer loop y")
        self.__plotter.plot_other_data_vs_time(index=[1, 2], save_plot=True)
        self.__plotter.set_title(f"Minimal {self.__derivative.name.lower()} trajectory with Drone {letter} values for roll, pitch and yaw")
        self.__plotter.add_other_data(other_data=data[: , 3], name=f"{letter} values for inner loop roll")
        self.__plotter.add_other_data(other_data=data[: , 4], name=f"{letter} values for inner loop pitch")
        self.__plotter.add_other_data(other_data=data[: , 5], name=f"{letter} values for inner loop yaw")
        self.__plotter.plot_other_data_vs_time(index=range(3, 6), save_plot=True)
        self.__plotter.reset()
        
    def get_drone_kd_values(self):
        return (read_data(f"drone_K_values_{self.__derivative.name}.csv"),
                read_data(f"drone_D_values_{self.__derivative.name}.csv"))


    def plot_2D_drone_path(self):
        self.__plotter.initialize(self.__optimal_trajectory, self.__rrt_waypoints, self.__durations, self.__derivative)
        self.__plotter.append_title_name("with rrt-star and drone path")
        self.__plotter.plot_2D_data(waypoint_label="rrt-star generated waypoints")
        self.__plotter.set_trajectory(self.__drone_path_oiac)
        self.__plotter.plot_2d_trajectory(label = "rrt-star drone path OIAC")
        self.__plotter.set_trajectory(self.__drone_path_pid)
        self.__plotter.plot_2d_trajectory(label = "rrt-star drone path PID")
        self.__plotter.set_waypoints(self.__main_waypoints)
        self.__plotter.plot_2d_waypoints(label='Waypoints')
        self.__plotter.display_labels_2d(save_plot = self.__save_plots)
        self.__plotter.reset()

    def plot_drone_force_and_torque(self):
        self.__plotter.initialize(self.__drone_force_and_torque_oiac, self.__rrt_waypoints, self.__durations, self.__derivative)
        oiac_force_n_torque, pid_force_n_torque = equalize_data_length(self.__drone_force_and_torque_oiac, self.__drone_force_and_torque_pid)
        self._rename_plot_drone_force_and_torque(
            oiac_force_n_torque, 3, pid_force_n_torque, ' trajectory: Throttle'
        )
        self.__plotter.plot_other_data_vs_time(index=range(2), ylabel="Force [N]")
        self._rename_plot_drone_force_and_torque(
            oiac_force_n_torque, 0, pid_force_n_torque, ' moments: Roll'
        )
        self.__plotter.plot_other_data_vs_time(index=range(2, 4), ylabel="Torque [Nm]")
        self._rename_plot_drone_force_and_torque(
            oiac_force_n_torque, 1, pid_force_n_torque, ' moments: Pitch'
        )
        self.__plotter.plot_other_data_vs_time(index=range(4, 6), ylabel="Torque [Nm]")
        self._rename_plot_drone_force_and_torque(
            oiac_force_n_torque, 2, pid_force_n_torque, ' moments: Yaw'
        )
        self.__plotter.plot_other_data_vs_time(index=range(6, 8), ylabel="Torque [Nm]")
        self.__plotter.reset()

    def _rename_plot_drone_force_and_torque(self, oiac_force_n_torque, arg1, pid_force_n_torque, arg3):
        self.__plotter.add_other_data(oiac_force_n_torque[:, arg1], "OIAC")
        self.__plotter.add_other_data(pid_force_n_torque[:, arg1], "PID")
        self.__plotter.set_title(f"Minimal {self.__derivative.name.lower()}{arg3}")

    def plot_distance_error(self):
        self.__plotter.initialize(self.__drone_path_oiac, self.__rrt_waypoints, self.__durations, self.__derivative)
        self.__plotter.set_title(f"Minimal {self.__derivative.name.lower()} trajectory vs drone path distance error OIAC")
        self.__plotter.plot_2D_distance_error(self.__optimal_trajectory, self.__drone_path_oiac, save_plot=self.__save_plots)
        self.__plotter.initialize(self.__drone_path_pid, self.__rrt_waypoints, self.__durations, self.__derivative)
        self.__plotter.set_title(f"Minimal {self.__derivative.name.lower()} trajectory vs drone path distance error PID")
        self.__plotter.plot_2D_distance_error(self.__optimal_trajectory, self.__drone_path_pid, save_plot=self.__save_plots)
        self.__plotter.reset()



    def plot_rrt(self):
        self.__rrtplotter.initialize(self.__rrt, self.__optimal_trajectory, self.__drone_path_oiac)
        self.__rrtplotter.set_rrt_path(self.__rrt_waypoints)
        self.__rrtplotter.plot_waypoints(self.__main_waypoints)
        self.__rrtplotter.plot_rrt_path()
        self.__rrtplotter.plot_obstacles()
        self.__rrtplotter.plot_optimal_trajectory()
        self.__rrtplotter.plot_executed_trajectory(color=(0,0,1))
        self.__rrtplotter.set_drone_trajectory(self.__drone_path_pid)
        self.__rrtplotter.plot_executed_trajectory(color=(1,1,1))
        self.__rrtplotter.display_and_save_plots(self.__save_plots, self.__derivative)
        self.__rrtplotter.reset()

    def build_optimal_trajectory(self, derivative: Derivative):
        self.__optimal_trajectory = self.generate_trajectory(self.__rrt_waypoints, self.__durations, derivative)
        save_optimal_trajectory(self.__optimal_trajectory, derivative)
        save_data("Durations.csv", self.__mintrajgen.get_durations())
    
    def plot_trajectory_only(self, derivative: Derivative):
        self.__plotter.initialize(self.__optimal_trajectory, self.__rrt_waypoints, self.__durations, derivative)
        self.__plotter.append_title_name("with rrt-star only")
        self.__plotter.set_3d_figure()
        self.__plotter.plot_3d_data(waypoint_label="rrt-star generated waypoints")
        self.__plotter.display_labels_3d(save_plot = False)
        self.__plotter.reset()
        self.__rrtplotter.initialize(self.__rrt, self.__optimal_trajectory, None)
        self.__rrtplotter.set_rrt_path(self.__rrt_waypoints)
        self.__rrtplotter.plot_waypoints(self.__main_waypoints)
        self.__rrtplotter.plot_rrt_path()
        self.__rrtplotter.plot_obstacles()
        self.__rrtplotter.plot_optimal_trajectory()
        self.__rrtplotter.display_and_save_plots(False, None, None)
        self.__rrtplotter.reset()

    def generate_trajectory(self, waypoints, durations, order):
        self.__mintrajgen.initialize(waypoints=waypoints, durations=durations, minimal_trajectory_derivative=order)
        self.__mintrajgen.set_maximum_velocity(0.1)
        self.__mintrajgen.set_dt(0.05)
        self.__mintrajgen.create_poly_matrices()
        self.__mintrajgen.compute_splines()
        return self.__mintrajgen.get_splines()


def find_and_print_KD_values(kval, dval):
    find_KD_minmaxmean("K throttle", kval, "K xy", "K rpy")
    find_KD_minmaxmean("D throttle", dval, "D xy", "D rpy")


def find_KD_minmaxmean(arg0, arg1, arg2, arg3):
    minmaxmean(arg0, arg1[:, 0])
    minmaxmean(arg2, arg1[:, 1:3])
    minmaxmean(arg3, arg1[:, 3:6])

def minmaxmean(name, data):
    min_data = np.min(data)
    max_data = np.max(data)
    avg_data = np.mean(data)
    with open(get_file_location(f'Drone {name} values.txt', 'resource/data'), 'w') as file:
        file.write(f"{name} Max: {max_data}\n")
        file.write(f"{name} Min: {min_data}\n")
        file.write(f"{name} Avg: {avg_data}\n")
    print(f"{name} Max: {max_data}")
    print(f"{name} Min: {min_data}")
    print(f"{name} Avg: {avg_data}")



def save_optimal_trajectory(trajectory: np.ndarray, order: Derivative):
    save_data(f"rrt_trajectory_{order.name}.csv", trajectory)

def main(): 
    runner = Runner()
    runner.set_main_waypoints(np.array([[2.48, -1.08, 1.0], [0.8, -2.0, 1.0], [-2.15, 0.03, 1.0], 
                                        [-1.18, 1.03, 1.0], [0.75, 1.95, 1.0], [1.73, 1.03, 1.0], 
                                        [0.8, 0.03, 1.0], [0.83, -0.95, 1.0], [2.48, -1.08, 1.0]]))
    runner.set_durations(read_data("Durations.csv").flatten())
    runner.build_rrt(run=False)
    #runner.build_optimal_trajectory(Derivative.JERK)
    #runner.plot_trajectory_only(Derivative.JERK)
    runner.plot_trajectory_and_drone_data(Derivative.JERK, save_plots=True)
    runner.plot_trajectory_and_drone_data(Derivative.SNAP, save_plots=True)



if __name__ == "__main__":
    main()
