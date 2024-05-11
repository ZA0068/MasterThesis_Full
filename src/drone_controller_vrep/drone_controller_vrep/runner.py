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

    #def plot_trajectory_and_drone_data(self, derivative: Derivative, save_plots: bool = False):
    #    self._set_plot_types(derivative, save_plots)
    #    self.__acquire_data()
    #    self.plot_all_3D()
    #    self.plot_KD_values(self.get_drone_kd_values())
    #    self.plot_2D_drone_path()
    #    self.plot_drone_force_and_torque()
    #    self.plot_distance_error_all()
    #    self.plot_rrt()

    def _set_plot_types(self, derivative, save_plots):
        self.__derivative = derivative
        self.__save_plots = save_plots

    
    def __acquire_data(self):
        self.__rrt_waypoints = read_data("rrt_path.csv")
        for diff in range(2):
            derivative = Derivative(diff + 3).name
            self.__derivatives.append(derivative)
            for gaussians in [0, 0.05, 0.1]:
                self.___acquire_data_1(derivative, gaussians)
            for con in range(2):
                controller = Controller(con).name
                for gaussians in [0, 0.05, 0.1]:
                    self.___acquire_data_2(derivative, controller, gaussians)
        self.__fix_data()
        
    def __fix_data(self):
        for data in self.__drone_trajectory:
            optimal_trajectory_length = self.__optimal_trajectory[data[0], data[2]].shape[0]
            drone_trajectory_length = self.__drone_trajectory[data].shape[0]
            if optimal_trajectory_length > drone_trajectory_length:
                diff = optimal_trajectory_length - drone_trajectory_length
                self.__drone_trajectory[data] = np.concatenate((self.__drone_trajectory[data], np.full((diff, self.__drone_trajectory[data].shape[1]), np.nan)))
    
    def ___acquire_data_2(self, derivative, controller, gaussians):
        append = self.append_text_gaussian(gaussians)
        self.__drone_trajectory[(derivative, controller, gaussians)] = read_data(f"drone_path_{derivative}_{controller}{append}.csv")
        self.__drone_force_and_torque[(derivative, controller, gaussians)] = read_data(f"drone_force_and_torque_{derivative}_{controller}{append}.csv")
        self.__drone_error_data[(derivative, controller, gaussians)] = read_data(f"drone_error_{derivative}_{controller}{append}.csv")

    def append_text_gaussian(self, gaussians):
        return f"_{gaussians}_gaussian" if gaussians != 0 else ""
        
    def ___acquire_data_1(self, derivative, gaussians):
        append = f"_{gaussians}_gaussian" if gaussians != 0 else ""
        self.__optimal_trajectory[(derivative, gaussians)] = read_data(f"rrt_trajectory_{derivative}{append}.csv")
        self.__drone_kd_values[0][(derivative, gaussians)] = read_data(f"drone_K_values_{derivative}{append}.csv")
        self.__drone_kd_values[1][(derivative, gaussians)] = read_data(f"drone_D_values_{derivative}{append}.csv")

    def plot_all_3D(self):
        plt.rcParams.update({'font.size': 60})
        fig = plt.figure(figsize=(20, 60))
        fig.suptitle("Optimal and Drone trajectories with RRT-star waypoints 3D plot", fontsize=45)
        gaussian_values = [0, 0.05, 0.1]
        plot_params = {
            "JERK": {"color": "lightblue", "linestyle": "-", "linewidth": 3},
            "SNAP": {"color": "lightgreen", "linestyle": "-", "linewidth": 3},
            "JERK_OIAC": {"color": "orange", "linestyle": "--", "linewidth": 8},
            "JERK_PID": {"color": "cyan", "linestyle": "dotted", "linewidth": 6},
            "SNAP_OIAC": {"color": "darkgreen", "linestyle": "--", "linewidth": 8},
            "SNAP_PID": {"color": "navy", "linestyle": "dotted", "linewidth": 6}
        }
        for index, gaussian in enumerate(gaussian_values):
            ax = fig.add_subplot(3, 1, index + 1, projection='3d')
            subtitle = f"with STD: {gaussian}" if gaussian != 0 else "without STD"
            ax.set_title(subtitle, fontsize=42)
            for key_suffix, params in plot_params.items():
                key = (key_suffix.split('_')[0], key_suffix.split('_')[-1], gaussian) if '_' in key_suffix else (key_suffix, gaussian)
                df = self.__optimal_trajectory.get(key, pd.DataFrame()) if 'OIAC' not in key_suffix and 'PID' not in key_suffix else self.__drone_trajectory.get(key, pd.DataFrame())
                ax.plot(df[:,0], df[:,1], df[:,2], label=f"{key_suffix.replace('_', ' by ')} trajectory", **params)
        plt.legend(fontsize=42)
        plt.tight_layout()
        if self.__save_plots:
            save_image("Optimal and Drone trajectories with RRT-star waypoints 3D plot")
        plt.show()

    def print_kd_terms(self):
        for derivative in self.__derivatives:
            for gaussian in [0, 0.05, 0.1]:
                kval = self.__drone_kd_values[0][derivative, gaussian]
                dval = self.__drone_kd_values[1][derivative, gaussian]
                find_and_print_KD_values(kval, dval, derivative, gaussian)
        

    def plot_KD_values(self):
            self.print_kd_terms()
            self.plot_KD_Terms()
            
    def plot_KD_Terms(self):
        self.__plotter.plot_KD_values(self.__drone_kd_values, self.__save_plots)
        
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
        self.__plotter.normal_error_plot(self.__drone_error_data)
        self.__plotter.error_bar_plot(self.__drone_error_data)

    def plot_rrt(self):
        self.__rrtplotter.initialize(self.__rrt, None, None)
        self.__rrtplotter.set_rrt_path(self.__rrt_waypoints)
        self.__rrtplotter.plot_rrt_path()
        self.__rrtplotter.plot_waypoints(self.__main_waypoints)
        self.__rrtplotter.plot_obstacles()
        colors = list(matplotlib.colors.cnames.keys())
        selected_color = random.sample(colors, 18)
        for i, optimal_trajectory in enumerate(self.__optimal_trajectory):
            self.__rrtplotter.set_optimal_trajectory(self.__optimal_trajectory[optimal_trajectory])
            self.__rrtplotter.plot_optimal_trajectory(color=selected_color[i])
        for j, drone_trajectory in enumerate(self.__drone_trajectory):
            self.__rrtplotter.set_drone_trajectory(self.__drone_trajectory[drone_trajectory])
            self.__rrtplotter.plot_executed_trajectory(color=selected_color[i+j+1])
        self.__rrtplotter.display_and_save_plots(self.__save_plots)

    def build_optimal_trajectory(self, derivative: Derivative):
        self.__optimal_trajectory = self.generate_trajectory(self.__rrt_waypoints, self.__durations, derivative)
        save_optimal_trajectory(self.__optimal_trajectory, derivative)
        save_data("Durations.csv", self.__mintrajgen.get_durations())
    
    def plot_trajectory_only(self, derivative: Derivative):
        self.__plotter.initialize(self.__optimal_trajectory, self.__rrt_waypoints, self.__durations, derivative)
        self.__plotter.set_title("Trajectory plot with RRT-star paths")
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
        plt.rcParams.update({'font.size': 60})
        fig = plt.figure(figsize=(20, 60))
        fig.suptitle("Optimal and Drone trajectories with RRT-star waypoints 2D plot", fontsize=45)
        gaussian_values = [0, 0.05, 0.1]
        plot_params = {
            "JERK": {"color": "lightblue", "linestyle": "-", "linewidth": 3},
            "SNAP": {"color": "lightgreen", "linestyle": "-", "linewidth": 3},
            "JERK_OIAC": {"color": "orange", "linestyle": "--", "linewidth": 8},
            "JERK_PID": {"color": "cyan", "linestyle": "dotted", "linewidth": 6},
            "SNAP_OIAC": {"color": "darkgreen", "linestyle": "--", "linewidth": 8},
            "SNAP_PID": {"color": "navy", "linestyle": "dotted", "linewidth": 6}
        }
        for index, gaussian in enumerate(gaussian_values):
            ax = fig.add_subplot(3, 1, index + 1)
            subtitle = f"with STD: {gaussian}" if gaussian != 0 else "without STD"
            ax.set_title(subtitle, fontsize=40)
            for key_suffix, params in plot_params.items():
                key = (key_suffix.split('_')[0], key_suffix.split('_')[-1], gaussian) if '_' in key_suffix else (key_suffix, gaussian)
                df = self.__optimal_trajectory.get(key, pd.DataFrame()) if 'OIAC' not in key_suffix and 'PID' not in key_suffix else self.__drone_trajectory.get(key, pd.DataFrame())
                ax.plot(df[:, 0], df[:, 1], label=f"{key_suffix.replace('_', ' by ')} trajectory", **params)
        plt.tight_layout()
        plt.legend(fontsize=40, loc='lower left')
        if self.__save_plots:
            save_image("Optimal and Drone trajectories with RRT-star waypoints 2D plot")
        plt.show()

    def plot_drone_force_and_torque_all(self):
        plt.rcParams.update({'font.size': 30})
        alpha_cycle = itertools.cycle([1, 0.75, 0.5])
        moment_colors = {'Roll': 'orange', 'Pitch': 'cyan', 'Yaw': 'red'}
        moment_styles = {'Roll': 'dotted', 'Pitch': 'dotted', 'Yaw': 'dotted'}

        # Define combinations of trajectory and control types
        combinations = [('JERK', 'OIAC'), ('JERK', 'PID'), ('SNAP', 'OIAC'), ('SNAP', 'PID')]
        stds = [0, 0.05, 0.1]  # List of standard deviations

        # Create a separate figure for each combination of trajectory type and control method
        for trajectory, control in combinations:
            fig, axes = plt.subplots(3, 2, figsize=(24, 20))  # Adjust size as needed
            title = f'Drone Force and Moments for {trajectory}-{control}'
            fig.suptitle(title)

            # Loop through each std value
            for row, std in enumerate(stds):
                # Retrieve the data for each combination and std
                data_key = (trajectory, control, std)
                data = self.__drone_force_and_torque[data_key]
                time_data = np.cumsum(np.ones_like(data[:, 0]) * 0.05)  # Generate time data assuming 0.05s intervals

                # First column for Throttle
                axes[row, 0].plot(time_data, data[:, 3], label='Throttle', color='blue', linestyle='dotted', linewidth=5)
                axes[row, 0].set_title(f'Throttle (STD: {std})')
                self._plot_drone_force_and_torque_all(
                    axes, row, 0, 'Force [N]'
                )
                # Second column for RPY
                axes[row, 1].set_title(f'RPY Moments (STD: {std})')
                for idx, moment in enumerate(['Roll', 'Pitch', 'Yaw']):
                    alpha = next(alpha_cycle)
                    axes[row, 1].plot(time_data, data[:, idx], label=f'{moment}', color=moment_colors[moment], linestyle=moment_styles[moment], linewidth=5, alpha=alpha)
                self._plot_drone_force_and_torque_all(
                    axes, row, 1, 'Moment [Nm]'
                )
            plt.tight_layout()
            if self.__save_plots:
                save_image(title)
            plt.show()

    def _plot_drone_force_and_torque_all(self, axes, row, arg2, arg3):
        axes[row, arg2].set_xlabel('Time [s]')
        axes[row, arg2].set_ylabel(arg3)
        axes[row, arg2].legend()

    def add_gaussian_noise(self, trajectory, std_dev):
        noise = np.random.normal(scale=std_dev, size=trajectory.shape)
        return trajectory + noise

        
    def plot_all_data_at_once(self, save_plots=False):
        self.__save_plots = save_plots
        self.__acquire_data()
        #self.plot_all_3D()
        #self.plot_KD_values()
        self.plot_all_2D()
        #self.plot_drone_force_and_torque_all()
        #self.plot_distance_error_all()
        #self.plot_rrt()


def main():
    pass
    #runner = Runner()
    #data_jerk = read_data("rrt_trajectory_JERK.csv")
    #data_snap = read_data("rrt_trajectory_SNAP.csv")
    #plotter = Plotter()
    #plotter.set_3d_figure()
    #data_jerk_gaussian_25 = runner.add_gaussian_noise(data_jerk, 0.05)
    #data_jerk_gaussian_50 = runner.add_gaussian_noise(data_jerk, 0.1)
    #plotter.plot_3d_trajectory(data_jerk_gaussian_25, label="Jerk trajectory with std 0.05", color="orange", linewidth=3)
    #plotter.display_labels_3d()
    #plotter.set_3d_figure()    
    #plotter.plot_3d_trajectory(data_jerk_gaussian_50, label="Jerk trajectory with std 0.1", color="red", linewidth=3)
    #plotter.display_labels_3d()
    #data_snap_gaussian_25 = runner.add_gaussian_noise(data_snap, 0.05)
    #data_snap_gaussian_50 = runner.add_gaussian_noise(data_snap, 0.1)
    #plotter.set_3d_figure()
    #plotter.plot_3d_trajectory(data_snap_gaussian_25, label="Snap trajectory with std 0.05", color="orange", linewidth=3)
    #plotter.display_labels_3d()
    #plotter.set_3d_figure()
    #plotter.plot_3d_trajectory(data_snap_gaussian_50, label="Snap trajectory with std 0.1", color="red", linewidth=3)
    #plotter.display_labels_3d()
    #
    #save_data("rrt_trajectory_JERK_0.05_gaussian.csv", data_jerk_gaussian_25)
    #save_data("rrt_trajectory_JERK_0.1_gaussian.csv", data_jerk_gaussian_50)
    #save_data("rrt_trajectory_SNAP_0.05_gaussian.csv", data_snap_gaussian_25)
    #save_data("rrt_trajectory_SNAP_0.1_gaussian.csv", data_snap_gaussian_50)

if __name__ == "__main__":
    main()