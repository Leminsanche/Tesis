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
numero_simulaciones = 375

for i in range(numero_simulaciones):

    if i < int(numero_simulaciones/3):
        print('################# Demiray ###########################')
        print(i)
        #rangos_constantes = [[0.02,0.1,16],[1,5,16]]
        a,b = random.uniform(0.02,0.4) , random.uniform(0.5,8)
        desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Demiray(a,b)
        constantes_utilizadas.append([a,b])
        modos_desp = np.loadtxt('SVD/Biaxial_Demiray/Modos_Desplazmaientos.txt')
        modos_grad = np.loadtxt('SVD/Biaxial_Demiray/Modos_Gradientes.txt')
        mlp.load_state_dict(torch.load("mlp_models_dis2grad/mlp_model_Demiray.pt",map_location=torch.device('cpu')))

    elif i >= int( numero_simulaciones/3) and i < int(2 * numero_simulaciones/3) :
        print('################# Yeoh ###########################')
        print(i)
        #rangos_constantes = [[0.001,0.03,8],[0.002,0.08,5],[0.002,0.01,9]]
        a,b,c = random.uniform(0.0005,0.04) , random.uniform(0.002,0.1) , random.uniform(0.001,0.03)
        desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Yeoh(a,b,c)
        constantes_utilizadas.append([a,b,c])
        modos_desp = np.loadtxt('SVD/Biaxial_Yeoh/Modos_Desplazmaientos.txt')
        modos_grad = np.loadtxt('SVD/Biaxial_Yeoh/Modos_Gradientes.txt')
        mlp.load_state_dict(torch.load("mlp_models_dis2grad/mlp_model_Yeoh.pt",map_location=torch.device('cpu')))

    elif i >= int(2* numero_simulaciones/3):
        print('################# Mooney ###########################')
        print(i)
        #rangos_constantes = [[0.1,0.9,16],[0,0,1],[3,14,16]]
        a,b,c = random.uniform(0.05,0.9) , random.uniform(0.1,0.2) , random.uniform(2,15)
        desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Mooney(a,b,c)
        constantes_utilizadas.append([a,b,c])
        modos_desp = np.loadtxt('SVD/Biaxial_Mooney/Modos_Desplazmaientos.txt')
        modos_grad = np.loadtxt('SVD/Biaxial_Mooney/Modos_Gradientes.txt')
        mlp.load_state_dict(torch.load("mlp_models_dis2grad/mlp_model_Mooney.pt",map_location=torch.device('cpu')))

        if desplazamientos.shape[0] != 31:
            continue

    

    mesh = pv.read(malla)
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
plt.savefig('Histograma_MSE_disp')



######################### Histograma error cuadratico medio ###############################
plt.hist(x = MSQRS)
plt.xlabel('Mean square error')
plt.savefig('Histograma_MSE')



######################### Histograma error relativo por simulacion ###############################
plt.hist(np.mean(Errores, axis = 0))
plt.xlabel('Error Relativo')
plt.savefig('Errores_Relativos')


######################### Archivo VTK error ###############################
mesh = pv.read(malla)
mesh.clear_data()
mesh['Errores nodales'] = np.mean(Errores, axis = 1)
mesh.set_active_scalars('Errores nodales')
#mesh.plot()
mesh.save('Grafico_errores.vtk')