import numpy as np
from vispy import scene, app
from vispy.scene import visuals
from copy import deepcopy
class Obstacle:
    
    def __init__(self, *args):
        self._init_args(*args)
        self.create_vertices()

    def _init_args(self, *args):
        match len(args):
            case 6:
                self._init_mode_args(args)
            case 2:
                self._init_mode_arrays(args)
            case 1:
                self._init_mode_list(args[0])
            case _:
                raise ValueError("Invalid argument: Arguments must be 6 parameters, 2x3 arrays or 1x6 array and with string as an optional specification")

    def _init_mode_arrays(self, args):
        match len(args[0]):
            case 6:
                self._init_mode_list(args[0], args[1])
            case 3:
                self._init_mode_arr(args)
            case _:
                raise ValueError("Invalid array sizes: Arrays must 2x3 arrays or 1x6 array")

    def _init_mode_arr(self, args):
        self._init_mode(args[0], args[1])

    def _init_mode_args(self, args):
        self._init_mode(args[:3], args[3:])

    def _init_mode_cross_arg(self, array):
        self._init_mode(array[::2], array[1::2])
        
    def _init_mode(self, array1, array2):
        self.set_origin(np.array(array1))
        self.set_dimensions(np.abs(np.array(array2) - self._origin))
        
    def _init_mode_list(self, array, mode='iiifff'):
        match mode:
            case 'iiifff':
                self._init_mode_args(array)
            case 'ififif':
                self._init_mode_cross_arg(array)
            case _:
                raise ValueError("Invalid mode")

    def inflate(self, scale):
        center = self._get_cuboid_center()
        new_vertices = (self._vertices - center) * scale + center
        new_obs = deepcopy(self)
        new_obs.set_dimensions(self._dimensions * scale)
        new_obs.create_vertices(new_vertices)
        return new_obs

    def create_vertices(self, vertices=None):
        if vertices is not None:
            self._vertices = vertices
        else:
            self._vertices = self._origin + self._corners() * self._dimensions

    def _corners(self):
        return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])

    def rotate(self, yaw=0, pitch=0, roll=0):
        angles = np.deg2rad([roll, pitch, yaw])
        sincos = self._calculate_rotation_parameters(angles)
        self._vertices = self._rotate_vertices(sincos)
        return self

    def _rotate_vertices(self, sincos):
        return np.dot(self._vertices - self._origin, self._get_rotation_matrix(sincos)) + self._origin

    def _calculate_rotation_parameters(self, angles):
        return np.hstack((np.sin(angles), np.cos(angles)))

    def _get_rotation_matrix(self, sincos):
        return np.array(
            [
                self.__row_1__(sincos),
                self.__row_2__(sincos),
                self.__row_3__(sincos),
            ]
        )

    def __row_3__(self, sincos):
        return [-sincos[1], sincos[3] * sincos[0], sincos[4] * sincos[3]]

    def __row_2__(self, sincos):
        return [sincos[3] * sincos[2], sincos[4] * sincos[5] + sincos[0] * sincos[1] * sincos[2], sincos[4] * sincos[1] * sincos[2] - sincos[5] * sincos[0]]

    def __row_1__(self, sincos):
        return [sincos[3] * sincos[5], sincos[5] * sincos[0] * sincos[1] - sincos[4] * sincos[2], sincos[4] * sincos[5] * sincos[1] + sincos[0] * sincos[2]]
    
    def move(self, x, y, z):
        self.set_origin(x, y, z)
        self.create_vertices()
        return self

    def set_origin(self, *xyz):
        self._origin = np.asarray(xyz).flatten()
        return self

    def set_dimensions(self, *dimensions):
        self._dimensions = np.asarray(dimensions).flatten()
        return self

    def scale(self, x, y, z):
        self._dimensions = self._dimensions * (x, y, z).flatten()
        self.create_vertices()
        return self
    
    def resize(self, x_f, y_f, z_f):
        """Resize the cuboid and reapply transformations."""
        self._dimensions = np.array([x_f, y_f, z_f])
        self.create_vertices()
        return self
    def is_inside(self, *xyz):
        return not np.all(self._is_point_inside(xyz))

    def _is_point_inside(self, points):
        return np.any(self._compare_point_and_cuboid(points), axis=-1)

    def _compare_point_and_cuboid(self, points):
        return self._get_axis_vectors(self._get_direction_vector(points)) > self._dimensions

    def _get_axis_vectors(self, direction_vector):
        return self._calculate_vector_point(direction_vector, self._get_normalized_vectors())

    def _calculate_vector_point(self, d, val):
        return np.abs(d @ val) * 2

    def _get_direction_vector(self, points):
        return points - self._get_cuboid_center() 

    def _get_cuboid_center(self):
        return (self._vertices[0] + self._vertices[6]) / 2.0

    def _get_normalized_vectors(self):
        return (self._vertices[[1, 3, 4]] - self._vertices[0]) / self._dimensions

    def _get_point(self):
        return np.concatenate((self._vertices[0], self._vertices[6]))

    def _get_point_crossed(self):
        return np.array(self._vertices)[[0, 6], :].flatten(order='F')
    
    def get_vertices(self):
        return self._vertices

    def get_point(self, mode='iiifff'):
        match mode:
            case 'iiifff':
                return self._get_point()
            case 'ififif':
                return self._get_point_crossed()
            case _:
                raise ValueError("Invalid mode")

    def get_origin(self):
        return self._origin
    def get_dimensions(self):
        return self._dimensions

    def center(self):
        self._origin = self._get_cuboid_center()
        return self

    @classmethod
    def center_origin(cls, *args):
        return cls(args).center()
    

    
if __name__ == '__main__':
    # Create an instance of the Obstacle class
    obs = Obstacle([1, 1, 1, 4, 4, -4], 'iiifff').rotate(80,-180,0)
    # Setup VisPy canvas
    canvas = scene.SceneCanvas(keys='interactive', size=(1920, 1080), show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'  # Or use 'arcball' for alternative interaction

    # Vertices for cuboid
    vertices = obs.get_vertices()
    # Assuming the cuboid is represented as a line loop
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Side faces
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 3, 7], [0, 7, 4],  # Side face
        [1, 2, 6], [1, 6, 5],  # Opposite side face
        [0, 1, 5], [0, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6]   # Back face
    ])
    mesh = visuals.Mesh(vertices=vertices, faces=faces, color=(1, 0, 0, 1), parent=view.scene)
    for line in lines:
        line = visuals.Line(pos=vertices[line], color=(0, 0, 0, 1), parent=view.scene, width=2)

    # Optionally, add a marker at the origin for reference
    origin_visual = visuals.Markers()
    origin_visual.set_data(obs.get_origin()[np.newaxis, :], edge_color=None, face_color=(0, 1, 0), size=10)
    view.add(origin_visual)

    # Configure the camera to view the entire cuboid
    view.camera.set_range(x=(-3, 3), y=(-3, 3), z=(-3, 3))


    app.run()