import numpy as np
from launch.Funciones import ManejoDatos
import pyvista as pv


Ubicacion_caso = 'Casos/cubochico/'
nombre_archivo =  'cubo.msh'
texto = Ubicacion_caso+nombre_archivo
mesh = pv.read(texto)
mesh.clear_data()
COO = mesh.points


Ubicacion_salida = 'Resultados/cubochico/'
file =   Ubicacion_salida + 'Desplazamientos_Demiray_cubo_0.txt'
desplazamientos = np.loadtxt(file)

COO_def = []
mag_desp  = []
for i in range(len(desplazamientos[0])):
    desp = ManejoDatos(desplazamientos[:,i],3).ModoVector()
    mag = ManejoDatos(desplazamientos[:,i],3).Magnitud()
    COO_n = COO + desp 
    COO_def.append(COO_n)
    mag_desp.append(mag)
    
    

# Create a plotter object and set the scalars to the Z height
plotter = pv.Plotter(notebook=True, off_screen=True)
plotter.add_mesh(
    mesh,
    scalars=ManejoDatos(desplazamientos[:,-1],3).Magnitud(),
    lighting=False,
    show_edges=True,
    scalar_bar_args={"title": "Magnitud Desplazamiento"}
)

grid = mesh

# Open a gif
plotter.open_gif("cubo.gif")
#plotter.camera_position='yz'
pts = grid.points.copy()

# Update Z and write a frame for each updated position
nframe = 15

for i in range(len(desplazamientos[0])):
    plotter.update_coordinates(COO_def[i], render=False)
    plotter.update_scalars(np.array(ManejoDatos(desplazamientos[:,i],3).Magnitud()), render=False)

    # Write a frame. This triggers a render.
    plotter.write_frame()

# Closes and finalizes movie
plotter.close()
