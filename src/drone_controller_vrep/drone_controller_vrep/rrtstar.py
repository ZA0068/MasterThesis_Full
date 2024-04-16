import copy
import time
import numpy as np
from scipy.spatial import KDTree
from obstacle import Obstacle
from mayavi import mlab

class RRTStar:
    def __init__(self):
        self.__rounding_value = 2
        self.clear()
    
    def reset(self):
        self.__start = None
        self.__goal = None
        self.__previous_cost = np.inf
        self.__current_cost = np.inf
        self.__tree = {}
        self.__best_path = None
        self.__best_tree = None
        self.__new_node = None
        self.__neighbors = None
        self.__dynamic_it_counter = 0
        self.__has_rewired = False
        self.__all_nodes = []
        self.__kdtree = None


    def clear(self):
        self.__start = None
        self.__goal = None
        self.__max_step = 1
        self.__max_iterations = 1000
        self.__epsilon = 0.15
        self.__neighborhood_radius = 1.5
        self.__space_limits = None
        self.__obstacles = []
        self.__previous_cost = np.inf
        self.__current_cost = np.inf
        self.__tree = {}
        self.__best_path = None
        self.__best_tree = None
        self.__new_node = None
        self.__neighbors = None
        self.__t = np.linspace(0, 1, 100)
        self.__dynamic_it_counter = 0
        self.__dynamic_break_at = round(self.__max_iterations / 10)
        self.__has_rewired = False
        self.__all_nodes = []
        self.__kdtree = None

    def initialize(self, start, goal, max_step, boundary, iterations, epsilon, cuboid_distance):
        self.set_start_and_goal(start, goal)
        self.set_max_step(max_step)
        self.set_max_iterations(iterations)
        self.set_epsilon(epsilon)
        self.set_cuboid_dist(cuboid_distance)
        self.set_boundaries(boundary)
        self._init_cost()
        self._init_nodes_and_tree()
        self._init_dynamic_counter()

    def _init_cost(self):
        self.__previous_cost = np.inf
        self.__current_cost = np.inf

    def _init_nodes_and_tree(self):
        self.__tree = {}
        self.__best_path = None
        self.__best_tree = None
        self.__new_node = None
        self.__neighbors = None
        self.__t = np.linspace(0, 1, 100)
        

    def _init_dynamic_counter(self):
        self.__dynamic_it_counter = 0
        self.__dynamic_break_at = round(self.__max_iterations / 10)
        self.__has_rewired = False
        self.__obstacles = []


    def set_cuboid_dist(self, cuboid_dist):
        self.__neighborhood_radius = cuboid_dist * self.__max_step

    def set_epsilon(self, ε):
        self.__epsilon = ε

    def _round_value(self, value):
        return value.round(self.__rounding_value)
    
    def set_start_and_goal(self, start, goal):
        if start is not None and goal is not None:
            self.__start = self._round_value(np.asarray(start))
            self.__kdtree = KDTree([self.__start])
            self.__all_nodes = [self.__start]
            self.__goal = self._round_value(np.asarray(goal))

    def set_max_step(self, max_step):
        self.__max_step = max_step

    def set_max_iterations(self, max_iterations):
        self.__max_iterations = max_iterations

    def add_obstacles(self, *obstacles):
        if obstacles is None:
            return
        if len(obstacles) == 1:
            obstacles = obstacles[0]
        for obstacle in obstacles:
            self.__obstacles.append(obstacle)

    def set_boundaries(self, *boundary):
        self.__space_limits = np.asarray(boundary).flatten()

    def _extract_boundaries(self, boundary):
        match len(boundary):
            case 2:
                lower, upper = boundary
            case 6:
                lower = boundary[:3]
                upper = boundary[3:]
            case _:
                raise ValueError("Boundaries must be a list of two elements: lower and upper limits.")
        return lower,upper

    def run(self):
        for it in range(self.__max_iterations):
            self._generate_new_node_with_neighbors()
            if not self.__neighbors: continue
            self._update_tree_and_rewire()
            if not self._is_path_found(self.__tree): continue
            self._find_best_tree_and_cost(it)
            if self.__dynamic_it_counter >= self.__dynamic_break_at: break
        self._finalize_path_search()

    def _find_best_tree_and_cost(self, it):
        self.get_path(self.__tree)
        self._check_rewiring()
        self._update_cost_and_store_best_tree(it)

    def _finalize_path_search(self):
        self._validate_best_path()
        self.__best_path = self.get_path(self.__best_tree)
        print(f"\nBest path found with cost: {self.__current_cost}")

    def _check_rewiring(self):
        if self.__has_rewired and self.__current_cost > self.__previous_cost:  # sanity check
                raise Exception("Cost increased after rewiring")

    def _validate_best_path(self):
        if not self._is_path_found(self.__best_tree):
            raise Exception("No path found")

    def _update_cost_and_store_best_tree(self, it):
        if self.__current_cost < self.__previous_cost:
            self._update_best_tree(it)
        else:
            self._update_dynamic_progress()

    def _update_dynamic_progress(self):
        self.__dynamic_it_counter += 1
        print(f"\r Percentage to stop unless better path is found: {self._calculate_progress_percentage()}%",
                        end="\t",
                    )

    def _calculate_progress_percentage(self):
        return np.round(self.__dynamic_it_counter / self.__dynamic_break_at * 100, self.__rounding_value)

    def _update_best_tree(self, it):
        print(f"Iteration: {it} | Cost: {self.__current_cost}")
        self.store_best_tree()
        self.__previous_cost = self.__current_cost
        self.__dynamic_it_counter = 0

    def _update_tree_and_rewire(self):
        self._update_tree()
        self.__has_rewired = self._rewire_safely()

    def _generate_new_node_with_neighbors(self):
        self.__new_node = self._generate_random_node()
        self.__new_node = self._adapt_random_node_position(self._find_nearest_node())
        self.__neighbors = self._find_valid_neighbors()

    def store_best_tree(self):
        self.__best_tree = copy.deepcopy(self.__tree)

    @staticmethod
    def path_cost(path):
        return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

    def _generate_random_node(self):
        if np.random.uniform(0, 1) < self.__epsilon:
            return self.__goal
        return self._round_value(self._generate_random_3D_point())

    def _generate_random_3D_point(self):
        return np.random.uniform(self.__space_limits[:3], self.__space_limits[3:])

    def _find_nearest_node(self):
        dist, idx = self.__kdtree.query(self.__new_node)
        return self.__all_nodes[idx], dist

    def _adapt_random_node_position(self, nearest_node):
        return self._round_value(self._interpolate_node(nearest_node[0], nearest_node[1]))
    
    def _interpolate_node(self, nearest_node, distance_nearest):
        if distance_nearest < self.__max_step:
            return self.__new_node
        return nearest_node + (self.__new_node - nearest_node) * self.__max_step / distance_nearest

    def _calculate_distance(self, first_node, second_node):
        return np.linalg.norm(first_node - second_node, axis=-1)

    def _find_valid_neighbors(self):
        return [
            self.__all_nodes[idx]
            for idx in self.__kdtree.query_ball_point(self.__new_node, self.__neighborhood_radius)
            if self._is_valid_connection(self.__all_nodes[idx], self.__new_node)
        ]

    def _find_best_neighbor(self):
        costs = []
        for neighbor in self.__neighbors:
            cost = np.linalg.norm(neighbor - self.__start)
            costs.append(cost)
        return self._round_value(self.__neighbors[np.argmin(costs)])

    def _update_tree(self):
        self.__all_nodes.append(self.__new_node)
        self.__kdtree = KDTree(self.__all_nodes)
        self._store_if_not_equal(self._find_best_neighbor())

    def _store_if_not_equal(self, parent_node):
        if not np.array_equal(parent_node, self.__new_node):
            self._store_node_in_tree(parent_node)

    def _store_node_in_tree(self, node_parent):
        self.__tree[self._get_node_key_string(self.__new_node)] = node_parent

    def _get_node_key_string(self, node):
        return str(self._round_node_to_list(node))

    def _round_node_to_list(self, node):
        return self._round_value(node).tolist()

    def _rewire_safely(self):
        new_node_key = self._get_node_key_string(self.__new_node)
        for neighbor in self.__neighbors:
            neighbor_key = self._get_node_key_string(neighbor)
            if self._is_valid_connection_and_has_lower_cost(new_node_key, neighbor, neighbor_key):
                self.__tree[neighbor_key] = self.__new_node
                return True
        return False

    def _is_valid_connection_and_has_lower_cost(self, new_node_key, neighbor, neighbor_key):
        return self._are_keys_different_and_no_collision(new_node_key, neighbor, neighbor_key) and self._is_neighbor_cost_less(neighbor)

    def _are_keys_different_and_no_collision(self, new_node_key, neighbor, neighbor_key):
        return (neighbor_key != new_node_key or self._is_valid_connection(neighbor, self.__new_node))

    def _is_neighbor_cost_less(self, neighbor):
        return self.estimate_cost_to_neighbor(neighbor) < self._get_current_cost_local(neighbor)


    def estimate_cost_to_neighbor(self, neighbor):
        return np.linalg.norm(self.__new_node - self.__start) + np.linalg.norm(
            neighbor - self.__new_node
        )

    def _get_current_cost_local(self, neighbor):
        return np.linalg.norm(neighbor - self.__start)

    def _is_valid_connection(self, current_node, next_node):
        if not self.__obstacles:
            return True
        return self._has_collision_with_obstacles(self._calculate_intermediate_points(current_node, next_node))

    def _has_collision_with_obstacles(self, points):
        return not np.any([obstacle.inflate(1.5).is_inside(points) for obstacle in self.__obstacles])

    def _calculate_intermediate_points(self, current_node, next_node):
        return np.outer(self.__t, next_node - current_node) + current_node


    def _is_path_found(self, tree):
        if tree is None:
            return False
        return self._get_node_key_string(self.__goal) in tree

    def get_path(self, tree):
        path = self._build_path(tree, *self._get_goal_info())
        self.__current_cost = RRTStar.path_cost(path)
        return np.array(path[::-1]).reshape(-1, 3)

    def _build_path(self, tree, path, node, s_time):
        start_key, node_key = self._init_local_path_keys(node)
        while node_key != start_key:
            node_key = self._update_local_node_and_path(tree, path, s_time, node_key)
        return path

    def _init_local_path_keys(self, node):
        return self._get_node_key_string(self.__start), self._get_node_key_string(node)

    def _update_local_node_and_path(self, tree, path, s_time, node_key):
        node = tree[node_key]
        path.append(node)
        self._restart_if_elapsed_time_exceeded(s_time)
        return self._get_node_key_string(node)

    def _get_goal_info(self):
        return [self.__goal], self.__goal, time.time()

    def _restart_if_elapsed_time_exceeded(self, s_time):
        if time.time() - s_time > 5:
            print("Restarting...")
            self.run()

    @classmethod
    def init_RRTStar(cls, start, goal, max_step, max_iterations, boundary, obstacles):
        instance = cls()
        instance.initialize(start=start,goal=goal,max_step=max_step,iterations=max_iterations,boundary=boundary,epsilon=0.15,cuboid_distance= 1.5)
        instance.add_obstacles(*obstacles)
        return instance
    
    @classmethod
    def default_RRTSTAR(cls, start, goal, max_step):
        return cls(start, goal, max_step)
    
    def get_best_tree(self):
        return self.__best_tree

    def get_best_path(self):
        return self.__best_path

    def get_obstacles(self):
        return self.__obstacles

    def get_start_point(self):
        return self.__start

    def get_goal_point(self):
        return self.__goal

if __name__ == "__main__":

    start = [0, 0, 0]
    goal = [8, 8, -8]
    lower_bound = [-10, -10, -10]
    upper_bound = [10, 10, 10]
    step_size = 0.5

    ceiling = Obstacle([-10, 10, -10, 10, 9, 10], 'ififif')
    floor = Obstacle([-10, 10, -10, 10, -10, -9], 'ififif')

    obstacles = [
        ceiling,
        floor,
        Obstacle([1, 1, 1, 4, 4, -4], 'iiifff').rotate(80,-180,0),
        Obstacle([5, 8, 2, 5, 0, 5], 'ififif').rotate(130,20, 30),
        Obstacle([1, 3, 3, 5, 0, 5], 'ififif').rotate(-200,-70, 45),
        Obstacle([4, 8, 7, 9, 0, 5], 'ififif').center().rotate(30,-20,10),
    ]

    rrt = RRTStar.init_RRTStar(
        start=start,
        goal=goal,
        max_step=step_size,
        max_iterations=1000,
        boundary=[-10,-10,-10,10,10,10],
        obstacles=obstacles,
    )
    rrt.run()

    mlab.figure(size=(1920, 1080))
    # plot start and goal nodes in red and green
    mlab.points3d(start[0], start[1], start[2], color=(1, 0, 0), scale_factor=.2, resolution=60)
    mlab.points3d(goal[0], goal[1], goal[2], color=(0, 1, 0), scale_factor=.2, resolution=60)

    tree = rrt.get_best_tree()
    # Collect all the nodes and lines
    nodes = []
    lines = []
    import ast
    for node_key, parent in tree.items():
        node = np.array(ast.literal_eval(node_key))
        nodes.extend((node, parent))
        lines.append([node, parent])

    # Convert to numpy arrays for efficient indexing
    nodes = np.array(nodes)
    lines = np.array(lines)

    # Draw all the nodes at once
    mlab.points3d(nodes[:, 0], nodes[:, 1], nodes[:, 2], color=(0, 0, 1), scale_factor=.1)

    # Draw all the lines at once
    for line in lines:
        mlab.plot3d(line[:, 0], line[:, 1], line[:, 2], color=(0, 0, 0), tube_radius=0.01)

    # find the path from the start node to the goal node
    path = rrt.get_best_path()

    # Plot the paths
    mlab.plot3d(path[:, 0], path[:, 1], path[:, 2], color=(1, 1, 0), tube_radius=0.05)
    # Add axes
    mlab.axes()

    # Add orientation axes
    mlab.orientation_axes()
    mlab.show()



