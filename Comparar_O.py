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
import random



mlp = MLP(8,8)

Errores = []
constantes_utilizadas = []
MSQRS = []
MSQRS_desp = []
mesh = pv.read('Casos/Biaxial_Demiray/Biaxial3DB.msh')
numero_simulaciones = 270
#mlp.load_state_dict(torch.load("mlp_models_dis2grad/mlp_model.pt"))
mlp.load_state_dict(torch.load("mlp_models_dis2grad/mlp_model.pt",map_location=torch.device('cpu')))
modos_desp = np.loadtxt('SVD/Biaxial/Modos_Desplazmaientos.txt')
modos_grad = np.loadtxt('SVD/Biaxial/Modos_Gradientes.txt')
for i in range(numero_simulaciones):

    if i < int(numero_simulaciones/3):
        print('################# Demiray ###########################')
        a,b = random.uniform(0.02,0.1) , random.uniform(1,5)
        desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Demiray(a,b)
        constantes_utilizadas.append([a,b])
        #modos_desp = np.loadtxt('SVD/Biaxial/Modos_Desplazmaientos.txt')
        #modos_grad = np.loadtxt('SVD/Biaxial/Modos_Gradientes.txt')
        #mlp.load_state_dict(torch.load("mlp_model_Demiray.pt"))

    elif i >= int( numero_simulaciones/3) and i < int(2 * numero_simulaciones/3) :
        print('################# Yeoh ###########################')
        a,b,c = random.uniform(0.001,0.03) , random.uniform(0.002,0.08) , random.uniform(0.002,0.01)
        desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Yeoh(a,b,c)
        constantes_utilizadas.append([a,b,c])
        #modos_desp = np.loadtxt('SVD/Biaxial/Modos_Desplazmaientos.txt')
        #modos_grad = np.loadtxt('SVD/Biaxial/Modos_Gradientes.txt')
        #mlp.load_state_dict(torch.load("mlp_model_Yeoh.pt"))

    elif i >= int(2* numero_simulaciones/3):
        print('################# Mooney ###########################')
        a,b,c = random.uniform(0.1,0.9) , random.uniform(0,0) , random.uniform(3,14)
        desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Mooney(a,b,c)
        constantes_utilizadas.append([a,b,c])
        #modos_desp = np.loadtxt('SVD/Biaxial/Modos_Desplazmaientos.txt')
        #modos_grad = np.loadtxt('SVD/Biaxial/Modos_Gradientes.txt')
        #mlp.load_state_dict(torch.load("mlp_model_Mooney.pt"))
        if desplazamientos.shape[0] != 31:
            continue

    
    mesh.clear_data()

    D = np.zeros([3*len(desplazamientos[0]),len(desplazamientos)])
    ite = 0

    for jt, j in enumerate(desplazamientos):
        D[:,ite] = j.reshape(-1)
        ite = ite +1

    grad = gradientes[0]
    coef_desp = np.linalg.lstsq(modos_desp,D)
    coef_desp = coef_desp[0]

    mse_desp = mean_squared_error(D, np.matmul(modos_desp,coef_desp))
    MSQRS_desp.append(mse_desp)


    #print(coef_desp.T)
    coef_grad = mlp(torch.asarray(coef_desp.T).double()).detach().numpy()
    gradientes_ROM  = np.matmul(modos_grad,coef_grad.T)
    gradientes_ROM_magnitud  = np.array(ManejoDatos(gradientes_ROM[:,-1],9).Magnitud()) 

    gradientes_FEM  = ManejoDatos(grad[:,-1],9).Magnitud()
    ERROR = error_mesh(gradientes_FEM ,gradientes_ROM_magnitud)
    Errores.append(ERROR)
    
    mse = mean_squared_error(gradientes_FEM, gradientes_ROM_magnitud)
    MSQRS.append(mse)

Errores = np.array(Errores).T


######################### Histograma error cuadratico medio Desplazamientos SVD ###############################
plt.hist(x = MSQRS_desp)
plt.xlabel('Mean square error displacement')
plt.savefig('Histograma_MSE_disp_O')



######################### Histograma error cuadratico medio ###############################
plt.hist(x = MSQRS)
plt.xlabel('Mean square error')
plt.savefig('Histograma_MSE_O')



######################### Histograma error relativo por simulacion ###############################
plt.hist(np.mean(Errores, axis = 0))
plt.xlabel('Error Relativo')
plt.savefig('Errores_Relativos_O')


######################### Archivo VTK error ###############################
mesh = pv.read(malla)
mesh.clear_data()
mesh['Errores nodales'] = np.mean(Errores, axis = 1)
mesh.set_active_scalars('Errores nodales')
#mesh.plot()
mesh.save('Grafico_errores_O.vtk')