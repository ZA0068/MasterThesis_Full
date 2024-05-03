from runner import Runner

def main(): 
    runner = Runner()
    waypoints = [[2.48, -1.08, 1.0], [0.8, -2.0, 1.0], [-2.15, 0.03, 1.0], 
                 [-1.18, 1.03, 1.0], [0.75, 1.95, 1.0], [1.73, 1.03, 1.0], 
                 [0.8, 0.03, 1.0], [0.83, -0.95, 1.0], [2.48, -1.08, 1.0]]
    runner.set_main_waypoints(waypoints)
    runner.set_durations()
    runner.build_environment(False)
    runner.plot_all_data_at_once(save_plots=True)



if __name__ == "__main__":
    main()
