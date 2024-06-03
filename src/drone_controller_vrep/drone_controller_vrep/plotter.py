from header_file import *

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
    
    
    def set_title(self, title):
        self.__title = title
        
    def reset(self):
        self.__title = ""
        self.__ax_3d = None
        self.__ax_2d = None
    
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
            save_image(self.__title)

    def plot_2d_waypoints(self, **kwargs):
        plt.scatter(self.__waypoint_data[:, 0], self.__waypoint_data[:, 1], **kwargs)

    def plot_2d_trajectory(self, trajectory_data, **kwargs):
        plt.plot(trajectory_data[:, 0], trajectory_data[:, 1], **kwargs)

    def plot_3D(self, **kwargs):
        self.set_3d_figure()
        self.plot_3d_data(**kwargs)
        self.display_labels_3d(**kwargs)

    def set_3d_figure(self, ax):
        self.__ax_3d = ax
        self.__ax_3d.view_init(elev=40, azim=-30)
        self.__ax_3d.set_xlim([-3, 3])
        self.__ax_3d.set_ylim([-3, 3])
        self.__ax_3d.set_zlim([0, 3])

    def plot_3d_data(self, **kwargs):
        trajectory_label, waypoint_label = self._pop_multiple_keys(kwargs, ['trajectory_label', 'waypoint_label'], ['Trajectory', 'Waypoints'])
        plt.rcParams.update({'font.size': 24})
        self.plot_3d_trajectory(self.__trajectory_data, label=trajectory_label)
        self.plot_3d_waypoints(label=waypoint_label)

    def display_labels_3d(self, **kwargs):
        self.show_labels_3D_plots()
        self.save_plot(**kwargs)
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
        plt.show()

    def show_labels_3D_plots(self) :
        plt.rcParams.update({'font.size': 24})
        self.__ax_3d.set_xlabel("X-axis [m]")
        self.__ax_3d.set_ylabel("Y-axis [m]")
        self.__ax_3d.set_zlabel("Z-axis [m]")
        self.append_title_name("3D")
        self.__ax_3d.set_title(f"{self.__title}")
        self.__ax_3d.legend()
        


        
    def plot_3d_waypoints(self, **kwargs):
        self.__ax_3d.scatter(self.__waypoint_data[:, 0], self.__waypoint_data[:, 1], self.__waypoint_data[:, 2], **kwargs)

    def plot_3d_trajectory(self, trajectory_data, **kwargs):
        self.__ax_3d.plot(trajectory_data[:, 0], trajectory_data[:, 1], trajectory_data[:, 2], **kwargs)

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

    def plot_2D_trajectory_vs_time(self, time_data, trajectory_data, **kwargs):
        plt.plot(time_data, trajectory_data, **kwargs)

    def plot_2D_waypoint_vs_time(self, duration_data, waypoint_data, **kwargs):
        plt.scatter(duration_data, waypoint_data, **kwargs)


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
            


    def plot_time_series(self, df, columns, ax, label):
        for col in columns:
            linestyle = next(self.line_cycle)
            α = next(self.alpha_cycle)
            ax.plot(df['Time'], df[col], label=f'{col} - {label}', linestyle=linestyle, alpha=α, linewidth=6)
        ax.legend()

    def plot_KD_values(self, kd_data, save_plot=False):
        self.line_cycle = itertools.cycle(['-', '--', '-.', ':'])
        plt.rcParams.update({'font.size': 30})
        stds = [0, 0.05, 0.1]
        data_cols = {
            'throttle': ['throttle'],
            'xy': ['outer_loop_x', 'outer_loop_y'],
            'rpy': ['roll', 'pitch', 'yaw']
        }
        titles = ['Throttle', 'XY Loop', 'RPY']

        for cnt, letter in enumerate(['K', 'D']):
            fig, axs = plt.subplots(3, 3, figsize=(36, 36))  # 3x3 grid for each standard deviation and data type
            fig.suptitle(f"{letter} Values Analysis", fontsize=32)

            for std_index, std in enumerate(stds):
                for data_index, (key, cols) in enumerate(data_cols.items()):
                    self.alpha_cycle = itertools.cycle([1, 0.8, 0.6, 0.4, 0.2])
                    ax = axs[std_index, data_index]
                    ax.set_title(f"{titles[data_index]} - STD {std}")

                    for key, data in kd_data[cnt].items():
                        if key[1] == std and key[0] in ['JERK', 'SNAP']:
                            df = pd.DataFrame(data, columns=['throttle', 'outer_loop_x', 'outer_loop_y', 'roll', 'pitch', 'yaw'])
                            time_data = np.cumsum(np.ones_like(df['throttle']) * 0.05)
                            df['Time'] = time_data
                            self.plot_time_series(df, cols, ax, f"{key[0]}")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_plot:
                save_image(f'{letter} values analysis')
            plt.show()

    def normal_error_plot(self, drone_error_data):
        max_length = np.cumsum(np.ones_like(drone_error_data["JERK", "OIAC", 0]) * 0.05)
        plt.rcParams.update({'font.size': 36})
        data_list = Plotter.create_data_list(drone_error_data)  # Assuming this method returns a list of dictionaries
        df = pd.DataFrame(data_list)
        linewidths = {'OIAC': 8, 'MRAC': 6}
        linestyles = {'OIAC': 'dashed', 'MRAC': 'dotted'}
        colors = [
            "#00eeee", "#7fd60f", "#ffff00", "#ffaa00", "#ff00ff",
            "#ff0000", "#0000ff", "#00ff00", "#aff00f", "#23200f",
            "#601ff5", "#22afa5"
        ]

        # Assuming there are 3 standard deviations that have been used
        stds = sorted(df['STD'].unique())  # Sort to ensure the plots are in order

        # Create a figure with 3 subplots arranged vertically
        fig, axes = plt.subplots(len(stds), 1, figsize=(20, 48))  # Adjust height as needed for clarity
        plt.suptitle("Distance error plot for all drone trajectories", fontsize=50)
        for ax, std in zip(axes, stds):
            ax.set_xlim(0, max_length[-1])
            group = df[df['STD'] == std]
            for i, row in group.iterrows():
                time_data = np.cumsum(np.ones_like(row['Data']) * 0.05)
                label = f"{row['Trajectory']} and {row['Controller']}"
                color = colors[i % len(colors)]
                linewidth = linewidths[row['Controller']]
                linestyle = linestyles[row['Controller']]
                ax.plot(time_data, row['Data'], label=label, color=color, linewidth=linewidth, linestyle=linestyle)

            ax.legend()
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Error distance [m]")
            append = "No std" if std == 0 else f"With std {std}"
            ax.set_title(f"{append}")
        plt.tight_layout()
        save_image("Distance error plot for all drone trajectories")
        plt.show()
    
    @staticmethod
    def create_data_list(drone_error_data):
        data_list = []
        for key, values in drone_error_data.items():
            trajectory, controller, std = key
            value = values.flatten()
            data_list.append({
                    'Trajectory': trajectory,
                    'Controller': controller,
                    'STD': std,
                    'Data': value
            })
        return data_list
        

    def error_bar_plot(self, drone_error_data: dict):
        data = []
        labels = []
        stds = sorted({key[2] for key in drone_error_data})
        std_colors = {0: '#0000ff', 0.05: '#00ff00', 0.1: '#ff0000'}  # STD to color mapping
        std_hatches = {0: '/', 0.05: '.', 0.1: ''}  # Hatching patterns for STDs
        matplotlib.rcParams['hatch.linewidth'] = 5.0
        for label, selected_data in drone_error_data.items():
            mean = np.mean(selected_data)
            std = np.std(selected_data)
            data.append((label[0], label[1], label[2], mean, std))  # Tuple of label parts and data

        # Create DataFrame
        df = pd.DataFrame(data, columns=["Trajectory", "Controller", "STD", "mean", "std"])
        group_keys = df.groupby(['Controller', 'Trajectory']).groups.keys()  # Unique groups

        # Plot setup
        plt.rcParams.update({'font.size': 30})
        fig, ax = plt.subplots(figsize=(20, 12))
        barWidth = 0.25
        # Determine bar positions
        positions = list(range(len(group_keys)))

        # Plotting
        for idx, (controller, trajectory) in enumerate(group_keys):
            group_df = df[(df['Controller'] == controller) & (df['Trajectory'] == trajectory)]
            base_offset = 10  # Base offset in points for left and right bars
            max_offset = 100  # Maximum offset in points for middle bar

            for std_idx, std in enumerate(stds):
                sub_df = group_df[group_df['STD'] == std]
                if not sub_df.empty:
                    pos = [p + std_idx * barWidth for p in [positions[idx]]]
                    bar = ax.bar(pos, sub_df['mean'], yerr=sub_df['std'], color=std_colors[std],
                                 width=barWidth, capsize=5, label=f'STD {std}' if idx == 0 else "",
                                 hatch=std_hatches[std], edgecolor='white', linewidth=1)

                    # Overlay bars to enhance hatch visibility
                    if std_hatches[std]:
                        ax.bar(pos, sub_df['mean'], width=barWidth, color='none', 
                               edgecolor='white', linewidth=3, hatch=std_hatches[std])

                    # Adjust text offsets
                    for rect, (mean, stddev) in zip(bar, sub_df[['mean', 'std']].values):
                        height = rect.get_height()
                        offset = base_offset

                        # Determine if this is the middle bar and adjust offset based on neighbors
                        if std == 0.05:
                            left_height = group_df[group_df['STD'] == 0]['mean'].values[0] if 0 in stds else 0
                            right_height = group_df[group_df['STD'] == 0.1]['mean'].values[0] if 0.1 in stds else 0

                            if not (abs(height - left_height) > 0.5 or abs(height - right_height) > 0.5):
                                offset = max_offset  # Significant offset for middle bar

                        # Create transformations
                        base_transform = ax.transData
                        text_transform = mtransforms.offset_copy(base_transform, fig=fig, y=offset, units='points')

                        # Annotations for mean and std
                        ax.text(rect.get_x() + rect.get_width() / 2, height, f'$\\mu={mean:.2f}$',
                                ha='center', va='bottom', color='blue', fontsize=30,
                                transform=text_transform)

                        # Consistent offset between mean and std annotations
                        text_transform = mtransforms.offset_copy(base_transform, fig=fig, y=offset + 30, units='points')
                        ax.text(rect.get_x() + rect.get_width() / 2, height, f'$\\sigma={stddev:.2f}$', 
                                ha='center', va='bottom', color='darkgreen', fontsize=30,
                                transform=text_transform)

        # Add some formatting and labels
        ax.set_xlabel("Controllers")
        ax.set_ylabel("Error [m]")
        ax.set_title("Distance error bar plot for all drone trajectories")
        # Set the x-ticks to be the middle of the groups
        ax.set_xticks([p + barWidth for p in positions])
        ax.set_xticklabels([f'{c} and {t}' for c, t in group_keys])

        # Create legend & Show the plot
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right')
        plt.tight_layout()
        save_image("Distance error bar plot for all drone trajectories")
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
