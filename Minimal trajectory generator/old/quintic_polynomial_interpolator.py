import numpy as np
from vispy import app, scene

# Your vertices and faces remain the same
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

faces = np.array([
    [0, 1, 2], [0, 2, 3],
    [4, 5, 6], [4, 6, 7],
    [0, 3, 7], [0, 7, 4],
    [1, 2, 6], [1, 6, 5],
    [0, 1, 5], [0, 5, 4],
    [2, 3, 7], [2, 7, 6]
])
# Edges of the cube, defined by the start and end vertex indices
edges = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
    [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
    [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
])

# Assuming vertices, faces, and edges are defined as above...

canvas = scene.SceneCanvas(keys='interactive', size=(1920, 1080), show=True, config={'samples': 4})
view = canvas.central_widget.add_view()

# Create the mesh and add it to the view
mesh = scene.visuals.Mesh(vertices=vertices, faces=faces, color='cyan', parent=view.scene)

# Draw black borders (edges) around the cube
for edge in edges:
    # Fetch the start and end points for each edge
    start_point, end_point = vertices[edge]
    # Ensure the line positions are correctly formatted as a NumPy array
    line = scene.visuals.Line(pos=np.array([start_point, end_point], dtype=np.float32), color='black', width=2, parent=view.scene)

# Add points at the vertices of the cube
points = scene.visuals.Markers()
points.set_data(vertices, edge_color='black', face_color='red', size=10)
view.add(points)

# Configure the view camera to encompass the entire scene
view.camera = 'turntable'
view.camera.set_range((-2, 2), (-2, 2), (-2, 2))

if __name__ == '__main__':
    app.run()
