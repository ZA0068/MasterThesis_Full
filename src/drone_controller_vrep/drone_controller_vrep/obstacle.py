import numpy as np
from mayavi import mlab

class Obstacle:
    def __init__(self, x0, y0, z0, x_f, y_f, z_f):
        self.origin = np.array([x0, y0, z0])  # Cuboid's origin
        self.dimensions = np.array([x_f, y_f, z_f])  # Dimensions
        self.rotation = [0, 0, 0]  # Initialize rotation state
        self.create_vertices()  # Initial creation of vertices

    def __init__(self, array, mode='iiifff'):
        assert len(array) == 6, "Array must have 6 elements"
        if mode == 'iiifff':
            self.origin = np.array(array[:3])
            self.dimensions = np.array(array[3:])  # Dimensions
        elif mode =='ififif':
            self.origin = np.array(array[::2])
            self.dimensions = np.array(array[1::2])
        else:
            raise ValueError("Invalid mode")
        self.rotation = [0, 0, 0]  # Initialize rotation state
        self.create_vertices()  # Initial creation of vertices

    def create_vertices(self):
        """Recalculate vertices based on current state."""
        corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        self.vertices = self.origin + corners * self.dimensions
        self.apply_rotation()  # Reapply rotation to preserve orientation

    def apply_rotation(self):
        """Apply the stored rotation to vertices to preserve orientation."""
        if not any(self.rotation):
            return  # Skip if no rotation
        
        psi, theta, phi = np.deg2rad(self.rotation)  # Convert angles to radians
        cz, sz = np.cos(psi), np.sin(psi)
        cy, sy = np.cos(theta), np.sin(theta)
        cx, sx = np.cos(phi), np.sin(phi)
        
        # Rotation matrix (Z * Y * X)
        rotation_matrix = np.array([
            [cy*cz, cz*sx*sy-cx*sz, cx*cz*sy+sx*sz],
            [cy*sz, cx*cz + sx*sy*sz, cx*sy*sz-cz*sx],
            [-sy, cy*sx, cx*cy]
        ])
        
        # Adjust vertices around the geometric center for rotation
        center = self.origin + self.dimensions / 2
        self.vertices = np.dot(self.vertices - center, rotation_matrix) + center

    def rotate(self, yaw=0, pitch=0, roll=0):
        """Store the rotation and apply it."""
        self.rotation = [yaw, pitch, roll]
        self.apply_rotation()
        return self

    def move(self, x, y, z):
        """Move the cuboid and reapply transformations."""
        self.origin += np.array([x, y, z])
        self.create_vertices()
        return self

    def scale(self, x, y, z):
        """Scale the cuboid and reapply transformations."""
        self.dimensions *= np.array([x, y, z])
        self.create_vertices()
        return self
    
    def resize(self, x_f, y_f, z_f):
        """Resize the cuboid and reapply transformations."""
        self.dimensions = np.array([x_f, y_f, z_f])
        self.create_vertices()
        return self

if __name__ == '__main__':
    obs = Obstacle(0, 0, 0, 1, 1, 1)
    print("Original Vertices:\n", obs.vertices)
    obs.rotate(45, 0, 0).scale(2, 2, 2).move(5, 5, 1)
    print("Rotated and Resized Vertices:\n", obs.vertices)
    mlab.figure(size=(1920, 1080))
    mlab.points3d(*obs.vertices.T, scale_factor=0.1)
    mlab.axes()
    mlab.orientation_axes()
    mlab.show()

