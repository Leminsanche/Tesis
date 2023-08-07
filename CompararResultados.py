from launch.gradientes import Gradientes_nodales_Vulcan
from launch.vulcan_handler import VulcanHandler
from intervul.readpos import VulcanPosMesh
from launch.Funciones import *
from launch.Lanzadores import *
import numpy as np
import matplotlib.pyplot as plt
import time
import pyvista as pv
from sklearn.metrics import mean_squared_error
import torch
from launch.mlp_model import MLP


mean_desp = np.loadtxt('SVD/Biaxial/Mean_Desplazmaientos.txt')
mean_grad = np.loadtxt('SVD/Biaxial/Mean_Gradientes.txt')
modos_desp = np.loadtxt('SVD/Biaxial/Modos_Desplazmaientos.txt')
modos_grad = np.loadtxt('SVD/Biaxial/Modos_Gradientes.txt')


desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Mooney(0.9,0,14)

mesh = pv.read(malla)
mesh.clear_data()

######################################################################################### Desplazamientos ###################################################
print('####################### Desplazamientos #######################')
D = np.zeros([3*len(desplazamientos[0]),len(desplazamientos)])
ite = 0

for jt, j in enumerate(desplazamientos):
    D[:,ite] = j.reshape(-1)
    ite = ite +1

grad = gradientes[0]
#print(grad.shape)

###################################################### Sistema Rectangular ###################################################################################
coef_desp = np.linalg.lstsq(modos_desp,D)
coef_desp = coef_desp[0]

print('Difernecia promedio Reconstruccion y original',np.mean(D- np.matmul(modos_desp,coef_desp)))
mse = mean_squared_error(D, np.matmul(modos_desp,coef_desp))
print('Mean square Error',mse)
##############################################################Graficos##########################################################################################
#    desp_FEM = D[:,-1]
#    desp_ROM = np.matmul(modos_desp,coef_desp)[:,-1]
#
#    desp_mag_FEM = ManejoDatos(desp_FEM,3).Magnitud()
#    desp_mag_ROM = ManejoDatos(desp_ROM,3).Magnitud() #No es precisamente un ROM sino solo una reconstruccion con el espacio de baja dimensionalidad
#
#    mesh['Magnitud desplazamientos FEM'] = desp_mag_FEM 
#    mesh.set_active_scalars('Magnitud desplazamientos FEM')
#    mesh.plot()#
#
#    ERROR = error_mesh(desp_mag_ROM,desp_mag_FEM)
#
#    mesh['ERROR'] = ERROR
#    mesh.set_active_scalars('ERROR')
#    mesh.plot()
##############################################################################################################################################################
########################################################## Gradientes ########################################################################################
print('####################### Gradientes #######################')
############ Modelo ML
mlp = MLP(8,8)
mlp.load_state_dict(torch.load("launch/mlp_model.pt"))
coef_grad = mlp(torch.asarray(coef_desp.T).double()).detach().numpy()
#print(coef_grad.shape)
gradientes_ROM  = np.matmul(modos_grad,coef_grad.T)
gradientes_ROM_magnitud  = np.array(ManejoDatos(gradientes_ROM[:,-1],9).Magnitud()) 

gradientes_FEM  = ManejoDatos(grad[:,-1],9).Magnitud()
ERROR = error_mesh(gradientes_FEM ,gradientes_ROM_magnitud)
mesh['ERROR Relativo'] = ERROR
mesh.set_active_scalars('ERROR Relativo')
mesh.plot(show_edges = True)
mse = mean_squared_error(gradientes_FEM, gradientes_ROM_magnitud)
print('Mean square error:',mse)



