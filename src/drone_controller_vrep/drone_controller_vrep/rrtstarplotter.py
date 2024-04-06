import numpy as np
from vispy import scene
from vispy.scene import visuals, transforms
from vispy import app
from rrtstar import RRTStar
from obstacle import Obstacle

class RRTPlotter:
    def __init__(self, rrt: RRTStar = None, optimal_trajectory: np.ndarray = None, real_trajectory: np.ndarray = None):
        self._init_canvas()
        self.set_RRT(rrt)
        self.set_optimal_trajectory(optimal_trajectory)
        self.set_drone_trajectory(real_trajectory)
        self._init_faces()
        self._init_edges()

    def _init_edges(self):
        self.__edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0], 
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ])

    def _init_faces(self):
        self.__faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6]
        ])

    def _init_canvas(self):
        self.__canvas = scene.SceneCanvas(keys='interactive', size=(1920, 1080), show=True, config={'samples': 4})
        self.__view = self.__canvas.central_widget.add_view()
        self.__view.camera = 'arcball'
        self.__view.camera.set_range(x=(-2, 2), y=(-2, 2), z=(-2, 2))

    def set_drone_trajectory(self, real_trajectory):
        self.real_trajectory = real_trajectory

    def set_optimal_trajectory(self, optimal_trajectory):
        self.optimal_trajectory = optimal_trajectory

    def set_RRT(self, rrt: RRTStar):
        self.__start_and_goal = np.array([rrt.get_start_point(), rrt.get_goal_point()])
        self.__path = rrt.get_best_path()
        self.__tree = rrt.get_best_tree()
        self.__obstacles = rrt.get_obstacles()

    def animate_trajectory(self, visual, delay=60):
        if self.real_trajectory is None:
            raise ValueError("Drone (real) trajectory is not provided")
        
        index = 0
        def update(ev):
            nonlocal index
            if index < len(self.real_trajectory):
                coord = self.real_trajectory[index]
                visual.transform = transforms.STTransform(translate=coord[:3])
                index += 1
            else:
                index = 0  # Reset or stop the timer as needed

        timer = app.Timer(interval=delay / 1000.0, connect=update, start=True)

    def plot_executed_trajectory(self):
        if self.real_trajectory is not None:
            visuals.Line(pos=self.real_trajectory[:, :3], color=(0, 0, 1), parent=self.__view.scene, method='gl')

    def plot_start_and_goal(self):
        visuals.Markers(parent=self.__view.scene).set_data(self.__start_and_goal, edge_color=None, face_color=['red', 'green'], size=20)


    def plot_obstacles(self):
        for idx, obs in enumerate(self.__obstacles):
            vertices = obs.get_vertices()
            inflated_vertices = obs.inflate(1.1).get_vertices()
            if idx in [0, 1]:
                visuals.Mesh(vertices=vertices, faces=self.__faces, color=(0.1, 0.1, 0.1, 1), parent=self.__view.scene)
                visuals.Mesh(vertices=inflated_vertices, faces=self.__faces, color=(0.1, 0.1, 0.1, 0.1), parent=self.__view.scene)
            else:
                visuals.Mesh(vertices=vertices, faces=self.__faces, color=(1, 0, 0, 1), parent=self.__view.scene)
                visuals.Mesh(vertices=inflated_vertices, faces=self.__faces, color=(1, 1, 1, .5), parent=self.__view.scene)
            for line in self.__edges:
                line = visuals.Line(pos=vertices[line], color=(0.1, 0.1, 0.1, 1), parent=self.__view.scene, width=2)

    def plot_tree(self):
        for node, parent in self.__tree.items():
            if parent is not None:
                node = np.array(eval(node))
                visuals.Line(pos=np.array([node, parent]), color=(0, 1, 1, 0.5), parent=self.__view.scene, method='gl')
                
    def plot_path(self, color_path=(1, 1, 0, 1)):
        visuals.Line(pos=self.__path, color=color_path, parent=self.__view.scene, method='gl')
        visuals.Markers(parent=self.__view.scene).set_data(self.__path[1:-1], edge_color=None, face_color=['cyan'], size=20)


    def plot_trajectory(self, color_traj=(0, 1, 1, 1)):
        if self.optimal_trajectory is not None:
            visuals.Line(pos=self.optimal_trajectory[:, :3], color=color_traj, parent=self.__view.scene, method='gl')

# Example usage
if __name__ == '__main__':
    ceiling = Obstacle([-10, 10, -10, 10, 9, 10], 'ififif')
    floor = Obstacle([-10, 10, -10, 10, -10, -9], 'ififif')
    obstacles = [
        ceiling,
        floor,
        Obstacle([1, 1, 1, 4, 4, -4]).rotate(80,-180,0),
    ]
    rrt = RRTStar.init_RRTStar(
        start=[0, 0, 0],
        goal=[8, 8, -8],
        max_step=0.5,
        max_iterations=1000,
        boundary=[-10,-10,-10,10,10,10],
        obstacles=obstacles,
    )
    rrt.run()
    plotter = RRTPlotter(rrt)
    plotter.plot_path()
    plotter.plot_obstacles()
    plotter.plot_start_and_goal()
    app.run()
