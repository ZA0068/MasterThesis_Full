from mayavi import mlab
import numpy as np

# The vertices you provided
vertices = np.array([
    [-1.41421356, -1.11022302e-16, -1.0],
    [1.11022302e-16, -1.41421356, -1.0],
    [1.41421356, 1.11022302e-16, -1.0],
    [-1.11022302e-16, 1.41421356, -1.0],
    [-1.41421356, -1.11022302e-16, 1.0],
    [1.11022302e-16, -1.41421356, 1.0],
    [1.41421356, 1.11022302e-16, 1.0],
    [-1.11022302e-16, 1.41421356, 1.0]
])

# Faces defined by the indices of the vertices forming each triangle
# Assuming the cuboid is a rectangular prism and faces are rectangles made of two triangles
faces = np.array([
    [0, 1, 2], [0, 2, 3],  # Bottom face
    [4, 5, 6], [4, 6, 7],  # Top face
    [0, 3, 7], [0, 7, 4],  # Side face
    [1, 2, 6], [1, 6, 5],  # Opposite side face
    [0, 1, 5], [0, 5, 4],  # Front face
    [2, 3, 7], [2, 7, 6]   # Back face
])

# Sample points
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])
z = np.array([1, 3, 5, 7, 9])

# Plot the points: size and color are optional
points = mlab.points3d(x, y, z, scale_factor=0.5, color=(1, 0, 0))
mlab.show()
