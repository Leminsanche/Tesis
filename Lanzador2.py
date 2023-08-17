from launch.Lanzadores import *
import numpy as np
import pyvista as pv
from launch.Funciones import *
from launch.Energias import *
import random



def Demiray():
    a,b = random.uniform(0.02,0.1) , random.uniform(1,5)
    desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Demiray(a,b)
    grad = gradientes[0]
    return


def Yeoh():
    a,b,c = random.uniform(0.001,0.03) , random.uniform(0.002,0.08) , random.uniform(0.002,0.01)
    desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Yeoh(a,b,c)
    grad = gradientes[0]

    return 


if __name__ == '__main__':
    import sys 
    
    modelo = sys.argv[1]
    numero_simulaciones = sys.argv[2]
    if modelo == 'Demiray':
        Demiray()

    elif modelo == 'Yeoh':
        Yeoh()


