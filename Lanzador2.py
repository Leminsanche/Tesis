from launch.Lanzadores import *
import numpy as np
import pyvista as pv
from launch.Funciones import *
from launch.Energias import *
import random



def Demiray(step_save = None):
    a,b = random.uniform(0.02,0.1) , random.uniform(1,5)
    desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Demiray_20(a,b,penal = 200000, save_step = step_save)
    #desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Demiray(a,b, save_step = step_save)
    grad = gradientes[0]
    j_final = gradientes[1]
    return [desplazamientos], [tensiones], [grad], malla, j_final,(a,b)


def Yeoh(step_save = None):
    a,b,c = random.uniform(0.001,0.03) , random.uniform(0.002,0.08) , random.uniform(0.002,0.01)
    desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Yeoh_20(a,b,c,penal = 200000, save_step = step_save)
    #desplazamientos, tensiones, gradientes, malla = Vulcan().Biaxial_Yeoh(a,b,c, save_step = step_save)

    grad = gradientes[0]
    j_final = gradientes[1]
    return [desplazamientos], [tensiones], [grad], malla, j_final,(a,b)


if __name__ == '__main__':
    import sys 
    
    modelo = sys.argv[1]
    numero_simulaciones = int(sys.argv[2])

    
    if modelo == 'Demiray':
        Nombre_info_salida = '_Demiray20_Biaxial_'  ## Nombre caso EJ: _Tracion_Demiray_
        Ubicacion_salida = 'Resultados/Biaxial_Demiray_20/'
        it = 0
        constantes = []
        a = [i for i in range(0,1501,20)] #np.append(np.arange(0,1501,60),1501)
        #a = [i for i in range(0,51,5)]
        print(a)
        

        while it < numero_simulaciones:
            print(f'####################################### Simulacion {it+1}/{numero_simulaciones} #######################################')
            numero_guardado = it
            desplazamientos, tensiones, gradientes, malla, j_final, cons = Demiray(step_save=a)
            nombre_T = 'Tensiones' + Nombre_info_salida + str(numero_guardado) + '.npz' 
            nombre_D = 'Desplazamientos' + Nombre_info_salida  + str(numero_guardado) + '.npz'
            nombre_G = 'Gradientes' + Nombre_info_salida  + str(numero_guardado) + '.npz'
            #print(desplazamientos[0])
            if j_final > 1.02 or desplazamientos[0].shape[0] != 1501:
                continue
            else:
                print(gradientes[0].shape)
                save_result(nombre_T ,nombre_D,nombre_G,Ubicacion_salida,desplazamientos[0][a,:,:],tensiones[0][a,:,:],gradientes[0])
                constantes.append(cons)

            it = it+1
        texto = Ubicacion_salida + 'Constantes_usadas.txt'
        np.savetxt(texto , constantes)
       
 ###################################################################### Yeoh  ######################################################################

    elif modelo == 'Yeoh':
        Nombre_info_salida = '_Yeoh20_Biaxial_'  ## Nombre caso EJ: _Tracion_Demiray_
        Ubicacion_salida = 'Resultados/Biaxial_Yeoh_20/'
        it = 0
        constantes = []
        a = [i for i in range(0,1501,20)] #np.append(np.arange(0,1501,60),1501)
        #a = [i for i in range(0,51,5)]
        print(a)

        while it < numero_simulaciones:
            print(f'####################################### Simulacion {it+1}/{numero_simulaciones} #######################################')
            numero_guardado = it
            desplazamientos, tensiones, gradientes, malla, j_final, cons = Yeoh(step_save=a)
            nombre_T = 'Tensiones' + Nombre_info_salida + str(numero_guardado) + '.npz' 
            nombre_D = 'Desplazamientos' + Nombre_info_salida  + str(numero_guardado) + '.npz'
            nombre_G = 'Gradientes' + Nombre_info_salida  + str(numero_guardado) + '.npz'
            print(desplazamientos[0].shape[0] )
            if j_final > 1.02 or desplazamientos[0].shape[0] != 1501:
                continue
            else: 
                print(gradientes[0].shape)
                save_result(nombre_T ,nombre_D,nombre_G,Ubicacion_salida,desplazamientos[0][a,:,:],tensiones[0][a,:,:],gradientes[0])
                constantes.append(cons)

            it = it+1
        texto = Ubicacion_salida + 'Constantes_usadas.txt'
        np.savetxt(texto , constantes)
