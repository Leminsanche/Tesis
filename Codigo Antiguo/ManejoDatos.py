import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from launch.vulcan_handler import VulcanHandler
from launch.Funciones import *
from intervul.readpos import VulcanPosMesh
import os

def get_results(posFile): #posFile
    vulcanData = VulcanPosMesh(posFile,VulcanPosMesh.MECHANICAL)
    results = vulcanData.getAllResults()
    return results['displacement'][:,:,:], results['stress'][:,:,:]
    

class ManejoDatos():
    ## Clase para el manejo de resultados 
    ## La metodologia utilizada trabaja con snaptchos
    ## datos agrupados en formato columna
    def __init__(self,vector,dim):
        self.vector = vector
        self.dim = dim
        
    def ModoVector(self):
        vector_dim = self.vector.reshape((int(len(self.vector)/self.dim),-1)) 
        return vector_dim
    
    def Magnitud(self):
        vector_mag = [np.linalg.norm(i) for i in self.ModoVector()]
        return vector_mag
        
        
direccion_resultados = 'Resultados/Biaxial3D_Demiray/'
direccion_svd = 'SVD/Biaxial_Demiray/'

nombre_tensiones =  'Tensiones_Demiray_Biaxial_' ## txts con los tension FOM
nombre_desplazamientos = 'Desplazamientos_Demiray_Biaxial_' ## txt con los desp FOM
nombre_desp_modos = 'Modos_desplazamientos.txt' ##Guardado modos desplazamientos
nombre_tens_modos = ' Modos_Tensiones.txt' ##Guardado modos tensiones
nombre_coef_T = 'Coefeicientes_Tensiones.txt' ## Guardado coef POD 
nombre_coef_D = 'Coefeicientes_Desplazamientos.txt' ## Guardado coef POD


lst = os.listdir(direccion_resultados)
number_files = len(lst)
print('numero de archivos: ',number_files)

numero_txt = number_files/2


Datos_T = []
Datos_D = []

for i in range(int(numero_txt)):
    nombre_T = direccion_resultados + nombre_tensiones + str(i+1) + '.txt' 
    nombre_D = direccion_resultados + nombre_desplazamientos+ str(i+1) + '.txt'
    
    
    T = np.loadtxt(nombre_T)
    D = np.loadtxt(nombre_D) 

    Datos_T.append(T)
    Datos_D.append(D)
    
    
    
Desplazamientos = Datos_D[0]
Tensiones = Datos_T[0]


for i in Datos_D[1:]:
    Desplazamientos = np.concatenate((Desplazamientos,i),axis =1)
    
for i in Datos_T[1:]:
    Tensiones = np.concatenate((Tensiones,i),axis =1)
    
    
    
Desplazamientos = Datos_D[0]
Tensiones = Datos_T[0]


for i in Datos_D[1:]:
    Desplazamientos = np.concatenate((Desplazamientos,i),axis =1)
    
for i in Datos_T[1:]:
    Tensiones = np.concatenate((Tensiones,i),axis =1)
    
    
#Desplazamientos.shape
#Tensiones.shape

Ud,Sd,Vd = np.linalg.svd(Desplazamientos)

Ut,St,Vt = np.linalg.svd(Tensiones) 




AcumD = []
AcumT = []

for it,i in enumerate(St):
    aux = sum(St[:it])
    AcumT.append(aux)
    
for it,i in enumerate(Sd):
    aux = sum(Sd[:it])
    AcumD.append(aux)
    

fig, (ax1,ax2) = plt.subplots(2, 1, figsize = (10,10))  
ax1.plot(AcumT/sum(St),'o')
ax1.grid()
ax1.set_title('Tensiones')

ax2.plot(AcumD/sum(Sd),'o')
ax2.grid()
ax2.set_title('Desplazamientos')
plt.show()
    


## Tensiones
r = 5
B_D = Ud[:,:r]
print('Energía acumulada en Desplazamientos es de:',(AcumD[r]/sum(Sd))*100,'%')

B_T = Ut[:,:r]
print('Energía acumulada en Tensiones es de:',(AcumT[r]/sum(St))*100,'%')

#### Guardado de datos y modos mas improtantes
np.savetxt(direccion_svd + nombre_desp_modos ,B_D)
np.savetxt(direccion_svd + nombre_tens_modos ,B_T)

## Caluclo de Coeficientes reduccion de orden X = fhi A

St_red = np.zeros((r,r))
Vt_red = Vt[:r,:]



Sd_red = np.zeros((r,r))
Vd_red = Vd[:r,:]

###### Para tensiones
it = 0
for i in range(r):
    for j in range(r):
        if i == j:
            St_red[i,j] = St[it]
            it = it +1
            
###### Para desplazamientos
it = 0
for i in range(r):
    for j in range(r):
        if i == j:
            Sd_red[i,j] = Sd[it]
            it = it +1
            

#caso = 0
A_T = np.matmul(St_red,Vt_red)
A_D = np.matmul(Sd_red,Vd_red)
np.savetxt(direccion_svd + nombre_coef_T,A_T)
np.savetxt(direccion_svd + nombre_coef_D,A_D)
    
    
    
