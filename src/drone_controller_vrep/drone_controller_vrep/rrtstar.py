import copy
import time
import numpy as np
from obstacle import Obstacle
from mayavi import mlab

class RRTStar:
    def __init__(self, start, goal, max_step):
        self.rounding_value = 2
        self.reset()
        self.set_start_and_goal(start, goal)
        self.set_max_step(max_step)
    
    def reset(self):
        self.set_start_and_goal([0,0,0], [1,1,1])
        self.set_max_step(1)
        self.set_max_iterations(1000)
        self.set_epsilon(0.15)
        self.obstacles = []
        self._init_cost()

        self.set_cuboid_dist(1.5)
        self.all_nodes = [self.start]

        self._init_tree()
        self._init_dynamic_counter()

    def _init_cost(self):
        self.previous_cost = np.inf
        self.current_cost = np.inf

    def _init_tree(self):
        self.tree = {}
        self.best_path = None
        self.best_tree = None

    def _init_dynamic_counter(self):
        self.dynamic_it_counter = 0
        self.dynamic_break_at = self.max_iterations / 10
        self._has_rewired = False


    def set_cuboid_dist(self, cuboid_dist):
        self.neighborhood_radius = cuboid_dist * self.max_step

    def set_epsilon(self, ε):
        self.epsilon = ε

    def set_start_and_goal(self, start, goal):
        self.start = np.asarray(start).round(self.rounding_value)
        self.goal = np.asarray(goal).round(self.rounding_value)

    def set_max_step(self, max_step):
        self.max_step = max_step

    def set_max_iterations(self, max_iterations):
        self.max_iterations = max_iterations

    def add_obstacles(self, *obstacles):
        if obstacles is None:
            return
        if isinstance(obstacles, list):
            obstacles = obstacles[0]
        for obstacle in obstacles:
            self.obstacles.append(obstacle)

    def set_boundaries(self, *boundary):
        lower, upper = self._extract_boundaries(*boundary)
        self.space_limits_lw = np.asarray(lower).flatten()
        self.space_limits_up = np.asarray(upper).flatten()

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
        for it in range(self.max_iterations):
            new_node, neighbors = self._generate_new_node_with_neighbors()
            if len(neighbors) == 0: continue

            self._update_tree_and_rewire(new_node, neighbors)

            if self._is_path_found(self.tree):
                self.get_path(self.tree)

                if self._has_rewired and self.current_cost > self.previous_cost:  # sanity check
                    raise Exception("Cost increased after rewiring")

                self._update_cost_and_store_best_tree(it)

                if self.dynamic_it_counter >= self.dynamic_break_at:
                    break

        self._validate_best_path()
        self.best_path= self.get_path(self.best_tree)
        print(f"\nBest path found with cost: {self.current_cost}")

    def _validate_best_path(self):
        if not self._is_path_found(self.best_tree):
            raise Exception("No path found")

    def _update_cost_and_store_best_tree(self, it):
        if self.current_cost < self.previous_cost:
            self._update_best_tree(it)
        else:
            self._update_dynamic_progress()

    def _update_dynamic_progress(self):
        self.dynamic_it_counter += 1
        print(f"\r Percentage to stop unless better path is found: {np.round(self.dynamic_it_counter / self.dynamic_break_at * 100, self.rounding_value)}%",
                        end="\t",
                    )

    def _update_best_tree(self, it):
        print(f"Iteration: {it} | Cost: {self.current_cost}")
        self.store_best_tree()
        self.previous_cost = self.current_cost
        self.dynamic_it_counter = 0

    def _update_tree_and_rewire(self, new_node, neighbors):
        self._update_tree(self._find_best_neighbor(neighbors), new_node)
        self._has_rewired = self._rewire_safely(neighbors, new_node)

    def _generate_new_node_with_neighbors(self):
        new_node = self._generate_random_node()
        nearest_node = self._find_nearest_node(new_node)
        new_node = self._adapt_random_node_position(new_node, nearest_node)
        neighbors = self._find_valid_neighbors(new_node)
        return new_node, neighbors

    def store_best_tree(self):
        """
        Update the best tree with the current tree if the cost is lower
        """
        # deepcopy is very important here, otherwise it is just a reference. copy is enough for the
        # dictionary, but not for the numpy arrays (values of the dictionary) because they are mutable.
        self.best_tree = copy.deepcopy(self.tree)

    @staticmethod
    def path_cost(path):
        """
        Calculate the cost of the path
        """
        return sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))

    def _generate_random_node(self):
        # with probability epsilon, sample the goal
        if np.random.uniform(0, 1) < self.epsilon:
            return self.goal

        x_rand = np.random.uniform(self.space_limits_lw[0], self.space_limits_up[0])
        y_rand = np.random.uniform(self.space_limits_lw[1], self.space_limits_up[1])
        z_rand = np.random.uniform(self.space_limits_lw[2], self.space_limits_up[2])
        return np.round(np.array([x_rand, y_rand, z_rand]), self.rounding_value)

    def _find_nearest_node(self, new_node):
        distances = [np.linalg.norm(new_node - node) for node in self.all_nodes]
        return self.all_nodes[np.argmin(distances)]

    def _adapt_random_node_position(self, new_node, nearest_node):
        """
        Adapt the random node position if it is too far from the nearest node
        """
        distance_nearest = np.linalg.norm(new_node - nearest_node)
        if distance_nearest > self.max_step:
            new_node = nearest_node + (new_node - nearest_node) * self.max_step / distance_nearest
            new_node = np.round(new_node, self.rounding_value)
        return new_node

    def _find_valid_neighbors(self, new_node):
        neighbors = []
        for node in self.all_nodes:
            node_in_radius = np.linalg.norm(node - new_node) <= self.neighborhood_radius
            if node_in_radius and self._is_valid_connection(node, new_node):
                neighbors.append(node)
        return neighbors

    def _find_best_neighbor(self, neighbors):
        """
        Find the neighbor with the lowest cost. The cost is the distance from the start node to the neighbor
        """
        costs = []
        for neighbor in neighbors:
            cost = np.linalg.norm(neighbor - self.start)
            costs.append(cost)

        return neighbors[np.argmin(costs)]

    def _update_tree(self, node, new_node):
        """
        Update the tree with the new node
        """
        # add the new node to the list of all nodes
        self.all_nodes.append(new_node)

        # add the new node to the tree
        node_key = str(np.round(new_node, self.rounding_value).tolist())
        node_parent = np.round(node, self.rounding_value)

        if not np.array_equal(node_parent, new_node):
            self.tree[node_key] = node_parent

    def _rewire_safely(self, neighbors, new_node):
        """
        Among the neighbors (without the already linked neighbor), find if linking to the new node is better than the
        current parent (re-wire).
        """
        for neighbor in neighbors:
            if np.array_equal(neighbor, self.tree[str(np.round(new_node, self.rounding_value).tolist())]):
                # if the neighbor is already the parent of the new node, skip
                continue

            if self._is_valid_connection(neighbor, new_node):
                current_parent = self.tree[str(np.round(neighbor, self.rounding_value).tolist())]

                # cost to arrive to the neighbor
                current_cost = np.linalg.norm(neighbor - self.start)

                # cost to arrive to the neighbor through the new node
                potential_new_cost = np.linalg.norm(new_node - self.start) + np.linalg.norm(neighbor - new_node)

                if potential_new_cost < current_cost:
                    # if it is cheaper to arrive to the neighbor through the new node, re-wire (update the parent of the
                    # neighbor to the new node)
                    self.tree[str(np.round(neighbor, self.rounding_value).tolist())] = new_node
                    return True
        return False

    def _is_valid_connection(self, current_node, next_node):
        if self.obstacles is None:
            return True
        t = np.linspace(0, 1, 100)
        points = np.outer(t, next_node - current_node) + current_node
        return not any(
            obstacle.is_inside(points[:, 0], points[:, 1], points[:, 2])
            for obstacle in self.obstacles
        )


    def _is_path_found(self, tree):
        """
        Check if the goal node is in the tree as a child of another node
        """
        goal_node_key = str(self.goal.round(self.rounding_value).tolist())
        return goal_node_key in tree.keys()

    def get_path(self, tree):
        path = [self.goal]
        node = self.goal
        s_time = time.time()

        while not np.array_equal(node, self.start):
            node = tree[str(node.round(self.rounding_value).tolist())]
            path.append(node)

            self._restart_if_elapsed_time_exceeded(s_time)

        self.current_cost = RRTStar.path_cost(path)
        return np.array(path[::-1]).reshape(-1, 3)

    def _restart_if_elapsed_time_exceeded(self, s_time):
        if time.time() - s_time > 5:
            print("Restarting...")
            self.run()

    @classmethod
    def init_RRTStar(cls, start, goal, max_step, max_iterations, boundary, obstacles):
        instance = cls(start, goal, max_step)
        instance.set_max_iterations(max_iterations)
        instance.set_boundaries(boundary)
        instance.add_obstacles(*obstacles)
        return instance

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
        Obstacle([4, 6, 3, 5, 0, 5], 'ififif').center().rotate(30,20,10),
        Obstacle([5, 8, 2, 5, 0, 5], 'ififif').center().rotate(30,20, 0),
        Obstacle([1, 3, 3, 5, 0, 5], 'ififif').center().rotate(30,-20,0),
        Obstacle([4, 8, 7, 9, 0, 5], 'ififif').center().rotate(30,-20,10),
    ]
    
    rrt = RRTStar.init_RRTStar(
        start=start,
        goal=goal,
        max_step=step_size,
        max_iterations=1000,
        boundary=[lower_bound, upper_bound],
        obstacles=obstacles,
    )
    rrt.run()

    mlab.figure(size=(1920, 1080))
    # plot start and goal nodes in red and green
    mlab.points3d(start[0], start[1], start[2], color=(1, 0, 0), scale_factor=.2, resolution=60)
    mlab.points3d(goal[0], goal[1], goal[2], color=(0, 1, 0), scale_factor=.2, resolution=60)

    tree = rrt.best_tree
    for node, parent in tree.items():
       node = np.array(eval(node))
       # plot the nodes and connections between the nodes and their parents
       mlab.points3d(node[0], node[1], node[2], color=(0, 0, 1), scale_factor=.1)
       mlab.points3d(parent[0], parent[1], parent[2], color=(0, 0, 1), scale_factor=.1)
       mlab.plot3d([node[0], parent[0]], [node[1], parent[1]], [node[2], parent[2]], color=(0, 0, 0), tube_radius=0.01)


    # find the path from the start node to the goal node
    path = rrt.best_path

    # Plot the paths
    mlab.plot3d(path[:, 0], path[:, 1], path[:, 2], color=(1, 1, 0), tube_radius=0.05)
    # Add axes
    mlab.axes()

    # Add orientation axes
    mlab.orientation_axes()
    mlab.show()



