# Helper functions for debugging / visualizing PlasticineLab

import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np

def plot_target_density(target_density):
    # TODO: Could add support for taichi field rather than numpy ndarray
    data = target_density
    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape) + 1  # one more than cell counts
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)

    # Add the scalar field data
    grid.cell_data["values"] = data.flatten(order="F")

    
    p = pv.Plotter()
    p.add_mesh(grid.outline(), color="black", line_width=1)  # outside cube
    p.add_volume(grid, cmap="viridis", opacity="sigmoid")
    p.show_axes=True
    p.show()

    
    threshold = 0.0001  # adjust as needed
    voxels = data > threshold  # binary mask

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels, edgecolor='k')

    fig = plt.figure()
    plt.plot(np.unique(data))

    plt.show()