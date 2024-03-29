import numpy as np
from mayavi import mlab
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
        self._origin = np.array(args[0])
        self._dimensions = np.array(args[1])

    def _init_mode_args(self, args):
        self._origin = np.array(args[:3])
        self._dimensions = np.array(args[3:])
    
    def _init_mode_list(self, array, mode='iiifff'):
        match mode:
            case 'iiifff':
                self._init_mode_args(array)
            case 'ififif':
                self._init_mode_cross_arg(array)
            case _:
                raise ValueError("Invalid mode")

    def _init_mode_cross_arg(self, array):
        self._origin = np.array(array[::2])
        self._dimensions = np.array(array[1::2])

    def create_vertices(self):
        corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        self._vertices = self._origin + corners * self._dimensions

    def rotate(self, yaw=0, pitch=0, roll=0):
        angles = np.deg2rad([roll, pitch, yaw])
        sincos = self._calculate_rotation_parameters(angles)
        self._vertices = np.dot(self._vertices, self._get_rotation_matrix(sincos))
        return self

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
        """Move the cuboid and reapply transformations."""
        self._origin += np.array([x, y, z])
        self.create_vertices()
        return self

    def scale(self, x, y, z):
        """Scale the cuboid and reapply transformations."""
        self._dimensions *= np.array([x, y, z])
        self.create_vertices()
        return self
    
    def resize(self, x_f, y_f, z_f):
        """Resize the cuboid and reapply transformations."""
        self._dimensions = np.array([x_f, y_f, z_f])
        self.create_vertices()
        return self

    def is_inside(self, x, y, z):
        return np.any(self._is_point_inside(self._get_point_Vector(x, y, z)))

    def _get_point_Vector(self, x, y, z):
        return self._get_points_on_line(self._modify_array(x, y, z))

    def _modify_array(self, x, y, z):
        return np.broadcast_arrays(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z))

    def _get_points_on_line(self, xyz):
        return np.stack([xyz[0].ravel(), xyz[1].ravel(), xyz[2].ravel()], axis=-1)

    def _is_point_inside(self, points):
        return np.all(self._compare_point_and_cuboid(points), axis=1)

    def _compare_point_and_cuboid(self, points):
        return self._get_axis_vectors(self._get_direction_vector(points)) <= self._dimensions

    def _get_axis_vectors(self, d):
        return self._calculate_vector_point(d, self._get_normalized_vectors())

    def _calculate_vector_point(self, d, val):
        return np.abs(np.dot(d, val)) * 2

    def _get_direction_vector(self, points):
        return points - self._get_cuboid_center() 

    def _get_cuboid_center(self):
        return (self._vertices[0] + self._vertices[6]) / 2.0

    def _get_normalized_vectors(self):
        return (self._vertices[[1, 3, 4]] - self._vertices[0]) / self._dimensions[:3]

    def get_vertices(self):
        return self._vertices

    def get_origin(self):
        return self._origin

if __name__ == '__main__':
    obs = Obstacle(0, 0, 0, 1, 1, 1)
    print("Original Vertices:\n", obs.get_vertices())
    x = np.linspace(2, 1, 100)
    y = np.linspace(3, 0, 100)
    z = np.linspace(-1, 0, 100)
    print("Is inside:", obs.is_inside(0, 0, 0))
    print("Is inside:", obs.is_inside(0.25, 0.25, 0.25))
    print("Is inside:", obs.is_inside(x, 0.5, 0.5))
    print("Is inside:", obs.is_inside(x, 1.5, 1.5))
    print("Is inside:", obs.is_inside(x, y, z))
    obs.rotate(45, 0, 0)
    print("Rotated and Resized Vertices:\n", obs.get_vertices())
    x = np.linspace(-0.2, 0.4, 100)
    y = np.linspace(-0.2, 0.4, 100)
    z = np.linspace(0, 1.05, 100)
    print("Is inside:", obs.is_inside(0, 0, 0))
    print("Is inside:", obs.is_inside(0.2, 0.2, 0.2))
    mlab.figure(size=(1920, 1080))
    mlab.points3d(*obs.get_origin(), scale_factor=0.2, color=(0, 1, 0))
    mlab.points3d(*obs.get_vertices().T, scale_factor=0.1, color=(1, 0, 0))
    mlab.axes()
    mlab.orientation_axes()
    mlab.show()



