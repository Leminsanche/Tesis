import numpy as np
from scipy import linalg
#### Codigo paras realizar el QR pivoting en el ensayo biaxial ####

Direccion_file = 'SVD/Biaxial2/Modos_Desplazmaientos.txt'
Direccion_Salida = 'SVD/Biaxial2/'
Nombre_Salida = 'Matriz_C'

import daal4py as d4p
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
algo = d4p.pivoted_qr()
A = np.loadtxt(Direccion_file)[:,:8]
mesh = pv.read('Casos/Biaxial_Demiray/Biaxial3DB.msh')
mesh.clear_data()

aux = A.copy()

puntos_imp = []
for it,i in enumerate(mesh.points):
    if i[0] != 0 and i[1] != 0 and i[2] != 0 :
        puntos_imp.append(3*it)
        puntos_imp.append(3*it+1)


a = algo.compute(A[puntos_imp,:])
indices_base = list(np.array(puntos_imp)[list(a.permutationMatrix[0])])

mesh_samples = np.zeros(A.shape[0])

C = np.zeros((A.shape[1],A.shape[0]))

jt = 0
for it  in range(A.shape[0]):
    if it in indices_base:
        mesh_samples[it] = 1
        C[jt,it] = 1
        jt = jt+1



texto_indices = Direccion_Salida + 'indices_base' + '.txt'
texto_paraview = Direccion_Salida + 'indices_base' + '.vtk'
texto = Direccion_Salida + Nombre_Salida + '.npz'
mesh['Samples'] = mesh_samples.reshape((-1,3))
np.savez_compressed(texto,C)
np.savetxt(texto_indices, np.array(indices_base) )
mesh.save(texto_paraview)