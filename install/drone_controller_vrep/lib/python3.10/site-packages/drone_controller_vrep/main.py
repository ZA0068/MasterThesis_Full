from runner import Runner
from header_file import *
def main(): 

    runner = Runner()
    waypoints = [[2.48, -1.08, 1.0], [0.8, -2.0, 1.0], [-2.15, 0.03, 1.0], 
                 [-1.18, 1.03, 1.0], [0.75, 1.95, 1.0], [1.73, 1.03, 1.0], 
                 [0.8, 0.03, 1.0], [0.83, -0.95, 1.0], [2.48, -1.08, 1.0]]
    runner.set_main_waypoints(waypoints)
    runner.build_environment(False)
    runner.set_durations()
    runner.plot_all_data_at_once(save_plots=True)

    #runner = Runner()
    #waypoints = read_data("rrt_path.csv")
    #durations = np.array([4.346263, 14.916099, 12.814055, 14.413188,
    #             13.992141, 13.931619, 12.732243, 11.561142,
    #             13.441726, 13.656134, 9.804591, 1.972308, 4.840822]) / 1
    #runner.set_durations(durations)
    #runner.set_rrt_waypoints(waypoints)
    #runner.build_optimal_trajectory(Derivative.SNAP)
    #runner.plot_trajectory_only(Derivative.SNAP)

if __name__ == "__main__":
    main()
