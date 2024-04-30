from runner import Runner





def main(): 
    runner = Runner()
    waypoints = [[2.48, -1.08, 1.0], [0.8, -2.0, 1.0], [-2.15, 0.03, 1.0], 
                 [-1.18, 1.03, 1.0], [0.75, 1.95, 1.0], [1.73, 1.03, 1.0], 
                 [0.8, 0.03, 1.0], [0.83, -0.95, 1.0], [2.48, -1.08, 1.0]]
    runner.set_main_waypoints()
    runner.set_durations()
    runner.build_rrt(run=False)
    #runner.build_optimal_trajectory(Derivative.SNAP)
    #runner.plot_trajectory_only(Derivative.SNAP)
    runner.plot_all_data_at_once(save_plots=False)



if __name__ == "__main__":
    main()
