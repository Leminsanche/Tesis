import numpy as np 
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.interpolate import Rbf
from launch.vulcan_handler import VulcanHandler
from intervul.readpos import VulcanPosMesh
from launch.Funciones import *
import time

Direccion = 'Casos/Biaxial Demiray/'
malla = 'Biaxial3D.msh'
mesh = pv.read(Direccion+malla)

direccion_svd = 'SVD/Biaxial/'

#A_V = np.loadtxt(direccion_svd+ 'Coefeicientes_Desplazamientos.txt')
A_V = np.loadtxt(direccion_svd+ 'Coeficientes_Desplazmaientos.txt')
fhi_V = np.loadtxt(direccion_svd+ 'Modos_Desplazmaientos.txt')
#fhi_T = np.loadtxt(direccion_svd+ 'Modos_Desplazamientos.txt')

caso = 10 ##################################################################

Variable_POD = ManejoDatos(np.matmul(fhi_V,A_V[:,caso *  31+30]),3).Magnitud()
#Desplazamientos_POD =  ManejoDatos(np.matmul(fhi_D,A_D[:,caso *  31+30]),3).Magnitud()

mesh['Magnitud Gradientes POD 5 Modos'] = Variable_POD
mesh.set_active_scalars('Magnitud Gradientes POD 5 Modos')
mesh.plot()


########################## Constante utilizadas Demiray ##########################################
constantes = []
constantes1 = np.linspace(0.02,0.1,18)
constantes2 = np.linspace(1,5,18)

for i in constantes1:
    for j in constantes2:
        constantes.append([i,j])

########################## Constante utilizadas Demiray ##########################################

########################## Constante utilizadas Yeoh ##########################################
#constantes = []

#constantes1 = np.linspace(0.001,0.03,8)
#constantes2 = np.linspace(0.002,0.08,5)
#constantes3 = np.linspace(0.002,0.01,9)

#for i in constantes1:
#    for j in constantes2:
#    	for k in constantes3:
#           constantes.append([i,j,k])
        
########################## Constante utilizadas Yeoh ##########################################

#################### FEM #############################
