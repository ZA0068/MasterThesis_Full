from header_file import *
from plotter import Plotter
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
        self.__optimal_trajectory = {}
        self.__drone_trajectory = {}
        self.__drone_force_and_torque = {}
        self.__drone_kd_values = [{}, {}]
        self.__drone_error_data = {}
        self.__main_waypoints = None
        self.__durations = None
        self.__derivatives = []
        
    def set_main_waypoints(self, main_waypoints):
        self.__main_waypoints = np.asarray(main_waypoints)
        
    def set_durations(self, durations=None):
        if durations is not None:
            self.__durations = np.asarray(durations)
        else:
            self.__durations = read_data("durations.csv").flatten()
    
    def set_rrt(self, rrt: RRTStar):
        self.__rrt = rrt
    
    def set_rrt_waypoints(self, rrt_waypoints: np.ndarray):
        self.__rrt_waypoints = rrt_waypoints
    
    def set_optimal_trajectory(self, optimal_trajectory: np.ndarray):
        self.__optimal_trajectory = optimal_trajectory
        
    def set_derivative(self, derivative: Derivative):
        self.__derivative = derivative
    
    def set_controller(self, controller: Controller):
        self.__controller = controller

    def build_environment(self, create_rrt_waypoints=False):
        self._setup_the_environment()
        self._generate_rrt_waypoints(create_rrt_waypoints)

    def _setup_the_environment(self):
        self.__rrt.add_obstacles(self._setup_obstacles())
        self.__rrt.set_max_step(0.1)
        self.__rrt.set_boundaries([-3, -3, 0, 3, 3, 5])

    def _setup_obstacles(self):
        obstacle1 = Obstacle([-1.3, -1.4, 0.0, -0.3, -0.4, 1.0])
        obstacle2 = Obstacle([-0.65, 1.05, 0.0, 0.35, 2.05, 1.0]).rotate(25, 0, 0)
        floor = Obstacle([-2.5, -2.5, -0.1, 2.5, 2.5, 0.0])
        return obstacle1,obstacle2,floor

    def _generate_rrt_waypoints(self, create_rrt_waypoints):
        if create_rrt_waypoints:
            self.__rrt_waypoints = prune_array(self.__build_rrt())
            save_data("rrt_path.csv", self.__rrt_waypoints)
        else: 
            self.__rrt_waypoints = read_data("rrt_path.csv")

    def __build_rrt(self):
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
        self.__acquire_data()
        self.plot_all_3D()
        self.plot_KD_values(self.get_drone_kd_values())
        self.plot_2D_drone_path()
        self.plot_drone_force_and_torque()
        self.plot_distance_error_all()
        self.plot_rrt()

    def _set_plot_types(self, derivative, save_plots):
        self.__derivative = derivative
        self.__save_plots = save_plots

    
    def __acquire_data(self):
        self.__rrt_waypoints = read_data("rrt_path.csv")
        for diff in range(2):
            derivative = Derivative(diff + 3).name
            self.__derivatives.append(derivative)
            self.__optimal_trajectory[derivative] = read_data(f"rrt_trajectory_{derivative}.csv")
            self.__drone_kd_values[0][derivative] = read_data(f"drone_K_values_{derivative}.csv")
            self.__drone_kd_values[1][derivative] = read_data(f"drone_D_values_{derivative}.csv")
            for con in range(2):
                controller = Controller(con).name
                self.__drone_trajectory[(derivative, controller)] = read_data(f"drone_path_{derivative}_{controller}.csv")
                self.__drone_force_and_torque[(derivative, controller)] = read_data(f"drone_force_and_torque_{derivative}_{controller}.csv")    
                self.__drone_error_data[(derivative, controller)] = read_data(f"drone_error_{derivative}_{controller}.csv")


    def plot_all_3D(self):
        self.__plotter.set_title("Optimal and Drone trajectories with RRT* waypoints")
        self.__plotter.set_3d_figure()
        self.__plotter.plot_3d_trajectory(self.__optimal_trajectory["JERK"], 
                                          label="Minimal jerk trajectory", 
                                          color="blue",
                                          linewidth=2)
        self.__plotter.plot_3d_trajectory(self.__optimal_trajectory["SNAP"], 
                                          label="Minimal snap trajectory",
                                          color="green",
                                          linewidth=2)
        self.__plotter.plot_3d_trajectory(self.__drone_trajectory["JERK", "OIAC"], 
                                          label="Drone path by OIAC and minimal jerk trajectory",
                                          color="orange",
                                          linestyle="--",
                                          linewidth=6)
        self.__plotter.plot_3d_trajectory(self.__drone_trajectory["JERK", "PID"], 
                                          label="Drone path by PID and minimal jerk trajectory",
                                          color="cyan",
                                          linestyle="dotted",
                                          linewidth=4)
        self.__plotter.plot_3d_trajectory(self.__drone_trajectory["SNAP", "OIAC"], 
                                          label="Drone path by OIAC and minimal snap trajectory",
                                          color= "lime",
                                          linestyle="--",
                                          linewidth=6)
        self.__plotter.plot_3d_trajectory(self.__drone_trajectory["SNAP", "PID"], 
                                          label="Drone path by PID and minimal snap trajectory",
                                          color= "navy",
                                          linestyle="dotted",
                                          linewidth=4)
        self.__plotter.display_labels_3d(save_plot = self.__save_plots)

    def print_kd_terms(self):
        for derivative in self.__derivatives:
            kval = self.__drone_kd_values[0][derivative]
            dval = self.__drone_kd_values[1][derivative]
            find_and_print_KD_values(kval, dval, derivative)
        

    def plot_KD_values(self):
            self.print_kd_terms()
            self.plot_KD_Terms()
            
    def plot_KD_Terms(self):
        self.__plotter.plot_KD_values(self.__drone_kd_values)
        
    def get_drone_kd_values(self):
        return (read_data(f"drone_K_values_{self.__derivative.name}.csv"),
                read_data(f"drone_D_values_{self.__derivative.name}.csv"))


    def plot_2D_drone_path(self):
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
        self.__plotter.plot_KD_values(index=range(2), ylabel="Force [N]")
        self._rename_plot_drone_force_and_torque(
            oiac_force_n_torque, 0, pid_force_n_torque, ' moments: Roll'
        )
        self.__plotter.plot_KD_values(index=range(2, 4), ylabel="Torque [Nm]")
        self._rename_plot_drone_force_and_torque(
            oiac_force_n_torque, 1, pid_force_n_torque, ' moments: Pitch'
        )
        self.__plotter.plot_KD_values(index=range(4, 6), ylabel="Torque [Nm]")
        self._rename_plot_drone_force_and_torque(
            oiac_force_n_torque, 2, pid_force_n_torque, ' moments: Yaw'
        )
        self.__plotter.plot_KD_values(index=range(6, 8), ylabel="Torque [Nm]")
        self.__plotter.reset()

    def _rename_plot_drone_force_and_torque(self, oiac_force_n_torque, arg1, pid_force_n_torque, arg3):
        self.__plotter.add_other_data(oiac_force_n_torque[:, arg1], "OIAC")
        self.__plotter.add_other_data(pid_force_n_torque[:, arg1], "PID")
        self.__plotter.set_title(f"Minimal {self.__derivative.name.lower()}{arg3}")

    def plot_distance_error_all(self):
        #self.__plotter.normal_error_plot(self.__drone_error_data)
        self.__plotter.error_bar_plot(self.__drone_error_data)

    def plot_rrt(self):
        self.__rrtplotter.initialize(self.__rrt, None, None)
        self.__rrtplotter.set_rrt_path(self.__rrt_waypoints)
        self.__rrtplotter.plot_rrt_path()
        self.__rrtplotter.plot_waypoints(self.__main_waypoints)
        self.__rrtplotter.plot_obstacles()
        cnt = 0
        for optimal_trajectory in self.__optimal_trajectory:
            self.__rrtplotter.set_optimal_trajectory(self.__optimal_trajectory[optimal_trajectory])
            self.__rrtplotter.plot_optimal_trajectory(color_traj=(0 + cnt,0.5 + cnt,1, 1))
            cnt =+ 0.5
        cnt = 0
        for drone_trajectory in self.__drone_trajectory:
            self.__rrtplotter.set_drone_trajectory(self.__drone_trajectory[drone_trajectory])
            self.__rrtplotter.plot_executed_trajectory(color=(0.2 + cnt,0.15 + cnt*0.65,1-0.5*cnt))
            cnt =+ 0.1
        self.__rrtplotter.display_and_save_plots(self.__save_plots)

    def build_optimal_trajectory(self, derivative: Derivative):
        self.__optimal_trajectory = self.generate_trajectory(self.__rrt_waypoints, self.__durations, derivative)
        save_optimal_trajectory(self.__optimal_trajectory, derivative)
        save_data("Durations.csv", self.__mintrajgen.get_durations())
    
    def plot_trajectory_only(self, derivative: Derivative):
        self.__plotter.initialize(self.__optimal_trajectory, self.__rrt_waypoints, self.__durations, derivative)
        self.__plotter.set_title("Trajectory plot with RRT* paths")
        self.__plotter.plot_3d_data(waypoint_label="rrt-star generated waypoints")
        self.__plotter.display_labels_3d(save_plot = False)
        self.__plotter.reset()
        self.__rrtplotter.initialize(self.__rrt, self.__optimal_trajectory, None)
        self.__rrtplotter.set_rrt_path(self.__rrt_waypoints)
        self.__rrtplotter.plot_waypoints(self.__main_waypoints)
        self.__rrtplotter.plot_rrt_path()
        self.__rrtplotter.plot_obstacles()
        self.__rrtplotter.plot_optimal_trajectory()
        self.__rrtplotter.display_and_save_plots(False, None)
        self.__rrtplotter.reset()

    def generate_trajectory(self, waypoints, durations, order):
        self.__mintrajgen.initialize(waypoints=waypoints, durations=durations, minimal_trajectory_derivative=order)
        self.__mintrajgen.set_maximum_velocity(0.1)
        self.__mintrajgen.set_dt(0.05)
        self.__mintrajgen.create_poly_matrices()
        self.__mintrajgen.compute_splines()
        return self.__mintrajgen.get_splines()

    def plot_all_2D(self):
        plt.rcParams.update({'font.size': 24})
        plt.figure(figsize=(10*2, 8*2))
        self.__plotter.set_title("Optimal and Drone trajectories with RRT* waypoints")
        self.__plotter.plot_2d_trajectory(self.__optimal_trajectory["JERK"], 
                                          label="Minimal jerk trajectory", 
                                          color="blue",
                                          linewidth=2)
        self.__plotter.plot_2d_trajectory(self.__optimal_trajectory["SNAP"], 
                                          label="Minimal snap trajectory",
                                          color="green",
                                          linewidth=2)
        self.__plotter.plot_2d_trajectory(self.__drone_trajectory["JERK", "OIAC"], 
                                          label="Drone path by OIAC and minimal jerk trajectory",
                                          color="orange",
                                          linestyle="--",
                                          linewidth=6)
        self.__plotter.plot_2d_trajectory(self.__drone_trajectory["JERK", "PID"], 
                                          label="Drone path by PID and minimal jerk trajectory",
                                          color="cyan",
                                          linestyle="dotted",
                                          linewidth=4)
        self.__plotter.plot_2d_trajectory(self.__drone_trajectory["SNAP", "OIAC"], 
                                          label="Drone path by OIAC and minimal snap trajectory",
                                          color= "lime",
                                          linestyle="--",
                                          linewidth=6)
        self.__plotter.plot_2d_trajectory(self.__drone_trajectory["SNAP", "PID"], 
                                          label="Drone path by PID and minimal snap trajectory",
                                          color= "navy",
                                          linestyle="dotted",
                                          linewidth=4)
        self.__plotter.display_labels_2d(save_plot = self.__save_plots)

    def plot_drone_force_and_torque_all(self):
        plt.rcParams.update({'font.size': 24})
        moment_colors = {'Roll': 'orange', 'Pitch': 'cyan', 'Yaw': 'red'}
        moment_styles = {'Roll': '--', 'Pitch': ':', 'Yaw': '-.'}

        for data in self.__drone_force_and_torque:
            time_data = np.cumsum(np.ones_like(self.__drone_force_and_torque[data][:, 0]) * 0.05)

            # Plot throttle in a separate figure
            plt.figure(figsize=(20, 16))  # Adjust size as needed
            self.__plotter.plot_2D_trajectory_vs_time(time_data, 
                                                      self.__drone_force_and_torque[data][:, 3], 
                                                      label="Throttle",
                                                      color='turquoise',
                                                      linestyle='-')
            self._entitle_drone_force_and_torque_all(
                'Drone Force: Throttle (',
                data,
                "Force [N]",
                'Drone Force: Throttle (',
            )
            # Plot Roll, Pitch, and Yaw in a combined figure
            plt.figure(figsize=(20, 16))
            # Plot roll
            self.__plotter.plot_2D_trajectory_vs_time(time_data, 
                                                      self.__drone_force_and_torque[data][:, 0], 
                                                      label="Roll",
                                                      color=moment_colors['Roll'],
                                                      linestyle=moment_styles['Roll'])
            # Plot pitch
            self.__plotter.plot_2D_trajectory_vs_time(time_data, 
                                                      self.__drone_force_and_torque[data][:, 1], 
                                                      label="Pitch",
                                                      color=moment_colors['Pitch'],
                                                      linestyle=moment_styles['Pitch'])
            # Plot yaw
            self.__plotter.plot_2D_trajectory_vs_time(time_data, 
                                                      self.__drone_force_and_torque[data][:, 2], 
                                                      label="Yaw",
                                                      color=moment_colors['Yaw'],
                                                      linestyle=moment_styles['Yaw'])

            self._entitle_drone_force_and_torque_all(
                'Drone Moments: Roll, Pitch, and Yaw (',
                data,
                "Moment [Nm]",
                'Drone Moments: Roll, Pitch, Yaw (',
            )

    # TODO Rename this here and in `plot_drone_force_and_torque_all`
    def _entitle_drone_force_and_torque_all(self, arg0, data, arg2, arg3):
        plt.title(f"{arg0}{data[0]} derivative and {data[1]} controller)")
        plt.xlabel("Time [s]")
        plt.ylabel(arg2)
        plt.legend()
        save_image(f"{arg3}{data[0]} derivative and {data[1]} controller)")
        plt.show()

        
    def plot_all_data_at_once(self, save_plots=False):
        self.__save_plots = save_plots
        self.__acquire_data()
        self.plot_all_3D()
        self.plot_KD_values()
        self.plot_all_2D()
        self.plot_drone_force_and_torque_all()
        self.plot_distance_error_all()
        self.plot_rrt()