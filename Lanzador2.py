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

    return [desplazamientos], [tensiones], [gradientes], malla


def Yeoh():
    a,b,c = random.uniform(0.001,0.03) , random.uniform(0.002,0.08) , random.uniform(0.002,0.01)
    desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Yeoh(a,b,c)
    grad = gradientes[0]

    return [desplazamientos], [tensiones], [gradientes], malla


if __name__ == '__main__':
    import sys 
    
    modelo = sys.argv[1]
    numero_simulaciones = int(sys.argv[2])
    if modelo == 'Demiray':
        Nombre_info_salida = '_Demiray20_Biaxial_'  ## Nombre caso EJ: _Tracion_Demiray_
        Ubicacion_salida = 'Resultados/Biaxial_Demiray_20/'
        for i in range(numero_simulaciones):
            print(f'####################################### Simulacion {i+1}/{numero_simulaciones} #######################################')
            numero_guardado = i
            desplazamientos, tensiones, gradientes, malla = Demiray()
            nombre_T = 'Tensiones' + Nombre_info_salida + str(numero_guardado) + '.npz' 
            nombre_D = 'Desplazamientos' + Nombre_info_salida  + str(numero_guardado) + '.npz'
            nombre_G = 'Gradientes' + Nombre_info_salida  + str(numero_guardado) + '.npz'

            save_result(nombre_T ,nombre_D,nombre_G,Ubicacion_salida,desplazamientos,tensiones,gradientes)

    elif modelo == 'Yeoh':
        Nombre_info_salida = '_Yeoh20_Biaxial_'  ## Nombre caso EJ: _Tracion_Demiray_
        Ubicacion_salida = 'Resultados/Biaxial_Yeoh_20/'
        for i in range(numero_simulaciones):
            print(f'####################################### Simulacion {i+1}/{numero_simulaciones} #######################################')
            numero_guardado = i
            desplazamientos, tensiones, gradientes, malla = Demiray()
            nombre_T = 'Tensiones' + Nombre_info_salida + str(numero_guardado) + '.npz' 
            nombre_D = 'Desplazamientos' + Nombre_info_salida  + str(numero_guardado) + '.npz'
            nombre_G = 'Gradientes' + Nombre_info_salida  + str(numero_guardado) + '.npz'

            save_result(nombre_T ,nombre_D,nombre_G,Ubicacion_salida,desplazamientos,tensiones,gradientes)


