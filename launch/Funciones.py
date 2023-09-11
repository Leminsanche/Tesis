import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from launch.vulcan_handler import VulcanHandler
from launch.Funciones import *
from intervul.readpos import VulcanPosMesh
import pyvista as pv
import time
import os
from os import remove
import itertools
import glob


class Resultados_vulcan():
     def __init__(self, pos_file):
          self.posFile = pos_file

     def desplazamientos(self):
        vulcanData = VulcanPosMesh(self.posFile,VulcanPosMesh.MECHANICAL)
        results = vulcanData.getAllResults()
        return results['displacement'][:,:,:]
     
     def Tensor_Tensiones(self):
        vulcanData = VulcanPosMesh(self.posFile,VulcanPosMesh.MECHANICAL)
        results = vulcanData.getAllResults()
        return results['stress'][:,:,:]
     
     def Fuerzas(self):
        vulcanData = VulcanPosMesh(self.posFile,VulcanPosMesh.MECHANICAL)
        results = vulcanData.getAllResults()
        return results['reaction'][:,:,:]
     
     def incompresibilidad(self):
        vulcanData = VulcanPosMesh(self.posFile,VulcanPosMesh.MECHANICAL)
        results = vulcanData.getAllResults()
        return results['J'][:,:,:]
     
     def Resultado(self,name):
            vulcanData = VulcanPosMesh(self.posFile,VulcanPosMesh.MECHANICAL)
            results = vulcanData.getAllResults()
            return results[name][:,:,:]
     


def get_results(posFile): #posFile
    vulcanData = VulcanPosMesh(posFile,VulcanPosMesh.MECHANICAL)
    results = vulcanData.getAllResults()
    #for i in results:
    #     print(i)
    return results['displacement'][:,:,:], results['stress'][:,:,:]
    
    
def save_result(nombre_T ,nombre_D,nombre_G,Direccion,desplazamientos,estres,gradientes):
  
    # D = np.zeros([3*len(desplazamientos[0][0]),len(desplazamientos[0]) * len(desplazamientos)])
    #S = np.zeros([6*len(estres[0][0]),len(estres[0]) * len(estres)])

       #S = np.zeros([len(estres[0]),len(estres)])
    # print(D.shape)
    # ite = 0
    # for it,i in enumerate(desplazamientos):
    #     for jt, j in enumerate(i):
    #         D[:,ite] = j.reshape(-1)
    #         ite = ite +1

    x,y,z = desplazamientos.shape
    D = desplazamientos.transpose((1,2,0)).reshape((y*z,x))
    #print(D2 == D)

    x,y,z = estres.shape
    S = estres.transpose((1,2,0)).reshape((y*z,x))
   
    # ite = 0
    # for it,i in enumerate(estres):
    #     for jt, j in enumerate(i):
    #         S[:,ite] = j.reshape(-1)
    #         ite = ite +1
            
    # print(S == S2)
    #for G in gradientes[1:]:
    #    grad = np.concatenate((grad,G),axis =1) 

    file_tensiones = Direccion + nombre_T
    file_desplazamientos = Direccion + nombre_D
    file_gradientes = Direccion + nombre_G
    np.savez_compressed(file_desplazamientos,D)
    np.savez_compressed(file_gradientes,gradientes)
    np.savez_compressed(file_tensiones,S)
    print('Desplazamientos, Tensiones y gradiente de deformacion, guardadas en resultados')
   
    return 
    
    
    
class ManejoDatos():
    """
    Clase creada para manejar los datos obtenidos en simulaciones y en ROM
    se considera una forma de matriz columna para los resultados obtenidos 
    
    input
    Vector: np.array() ; EX: np.array([[X],[Y],[Z]])
            Modo Vector:  np.array([X,Y,Z])
            Magnitud: np.array([(X**2 + Y**2 + Z**2 )**0.5])
    """
    
    def __init__(self,vector,dim):
        self.vector = vector
        self.dim = dim
        
    def ModoVector(self):
        vector_dim = self.vector.reshape((int(len(self.vector)/self.dim),-1)) 
        return vector_dim
    
    def Magnitud(self):
        vector_mag = [np.linalg.norm(i) for i in self.ModoVector()]
        return vector_mag
        
        
        
def Borrar_file(lista):
	""" 
	lista: Lista con nombre de archivos para eliminar
	"""
	for i in lista:
		remove(i)
		
	print('Archivos temporales borrados')
	return 
	
	


def Combinaciones(Parametros_constates):
    """
    Fucion para generar la combinacion de constantes para las simulaciones
    Parametros_constante: lista con los valores que cambiaran las constantes [C1 , C2, C3, ...Ci] con Ci = [x_inicial, x_final, numero de puntos]
    esta funcion utiliza np.linspace para generar los puntos e intertools para hacer la convinaciones
    """
    aux = []
    for i in Parametros_constates:
        aux.append(np.linspace(i[0],i[1],i[2]))

    constantes = []
    for t in itertools.product(*aux):
        constantes.append(t)

    return constantes

class Errores():
    def __init__(self,FEM,ROM):
        self.FEM = FEM
        self.ROM = ROM 
        
    def error_relativo(self,texto = True):
        """
        Funcion para calcular el error en los nodos entre una simulación FEM y ROM
        Compara los valores en los nodos para una malla y calcula el error
        error = |(y_teo - y-exp)/y_teo|*100
        """
        errores = []
        for it,i in enumerate(self.FEM):
            y_teo = abs(self.FEM[it])
            y_exp = abs(self.ROM[it])

            if y_teo == 0:
                error = (abs((y_teo - y_exp)/(y_teo+1e-10)) )*100

            else: 
                error = (abs((y_teo - y_exp)/y_teo) )*100
            
            errores.append(error)

        if texto == True:
            print('Error Promedio', round(np.mean(errores),10),'%')
            
        return np.array(errores) 
    

    def Norma_L2(self,texto = True):
        L2  = np.linalg.norm(self.FOM - self.ROM)
        if texto == True:
            print('Norma L2 Promedio', round(np.mean(L2),10))
        return L2
    
    def residuo(self, texto = True):

        errores = []
        for it,i in enumerate(self.FEM):
            y_teo = abs(self.FEM[it])
            y_exp = abs(self.ROM[it])

            error = abs((y_teo - y_exp))
            errores.append(error)

        if texto == True:
            print('Residuo Promedio', np.mean(errores))
            
        return np.array(errores)


def Trapecio_discreto(x,y):
    I  = 0

    for it in range(len(y)-1):
        a,b,fa,fb = x[it] , x[it+1] , y[it] , y[it+1]
        I =  I + (b-a) * ((fa+fb)/(2)) 
    return I 





def files_vulcan(Ubicacion_caso):
    """
    Funcion dedicadsa a determinar el nombre de los archivos
    dat, fix, geo y msh, asociados a la simulacion de vulcan
    input:

    ubicacion_caso: Carpeta en la cual esta el caso
    dat,fix,geo,msh : direccion donde estas los archivos con sus respectivas extensiones
    """
    extensiones = ['*.dat', '*.fix', '*.geo', '*.msh']
    aux = []
    
    for ext in extensiones:
        archivos = glob.glob(Ubicacion_caso+ext)
    
        aux.append(archivos[0])

    dat,fix,geo,msh = aux
    return  dat,fix,geo,msh