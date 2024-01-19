#'../assets/objs/cube.obj'
import pyvista as pv
import numpy as np
# Load mesh and texture into PyVista
mesh_path = '../assets/objs/dragon.obj'
mesh = pv.read(mesh_path)

# Initialize the plotter object with four sub plots
pl = pv.Plotter()
# 
voxels = pv.voxelize(mesh, density=0.02, check_surface=False)

#for cell in voxels.cell:
#    print(cell.point_ids)
# We add the voxels as a new mesh, add color and show their edges
pl.add_mesh(voxels, color=True, show_edges=True)



# Link all four views so all cameras are moved at the same time
pl.link_views()
# Set camera start position
pl.camera_position = 'xy'
# Show everything
pl.show()