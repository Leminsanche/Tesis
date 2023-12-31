from launch.gradientes import Gradientes_nodales_Vulcan
from launch.vulcan_handler import VulcanHandler
from intervul.readpos import VulcanPosMesh
from launch.Funciones import *
import numpy as np
import matplotlib.pyplot as plt
import time
import pyvista as pv

inicio = time.time()


desplazamientos = []
estres = []
gradientes = []

##################### Rangos constantes ########################
# Las constantes Dependen del modelo constitutivo usado y el rango
# de valores que se dan
rangos_constantes = [[0.02,0.1,16],[1,5,16]]
#rangos_constantes = [[0.02,0.1,2],[1,5,2]]
constantes = Combinaciones(rangos_constantes)
        
Ubicacion_caso = 'Casos/'
Nombre_salida = 'File_vulcan'
Nombre_info_salida = '_Caso_'  ## Nombre caso EJ: _Tracion_Demiray_
Ubicacion_salida = 'Resultados/Carpeta/'

param = {'Cons1': 0.02,'Cons2': 1,'Penal': 20 }  

dat, geo, fix, file_msh = files_vulcan(Ubicacion_caso)
caso1 = VulcanHandler([dat,geo,fix],Nombre_salida)



contador = 0
numero_guardado = 0

for i in range(len(constantes)):

    print('######################## Simulacion ',i+1, 'de ',len(constantes),' ########################' )
    
    for jt,j in enumerate(param):
        if jt == (len(param) - 1):
            param[j] = constantes[i][0]*1000

            print(j , '=', param[j]) #000

        else:
            param[j] = constantes[i][jt]
            print(j , '=', constantes[i][jt] )
    #print(param)
    caso1.run(param)
    
    disp, stress = get_results(caso1.pathToPos())
    print('Numero de pasos simulados: ', len(disp))
    print('Simulacion Finalizada\n')
    print('Inicio calculo de Gradiantes de deformacion')
    gradientes_deformacion = Gradientes_nodales_Vulcan(file_msh,disp)
    #print('Final calculo de Gradiantes de deformacion')
    contador = contador +1 #Cuenta las simulacion que se han realizado
    desplazamientos.append(disp)
    estres.append(stress)
    gradientes.append(gradientes_deformacion)
    
    if contador == 1:    #Guarda cuando se realizan n simulaciones (contador == n)
        nombre_T = 'Tensiones' + Nombre_info_salida + str(numero_guardado) + '.npz' 
        nombre_D = 'Desplazamientos' + Nombre_info_salida  + str(numero_guardado) + '.npz'
        nombre_G = 'Gradientes' + Nombre_info_salida  + str(numero_guardado) + '.npz'
        #print(nombre_T ,nombre_D,nombre_G,Ubicacion_salida)
        save_result(nombre_T ,nombre_D,nombre_G,Ubicacion_salida,desplazamientos,estres,gradientes)
        
        desplazamientos = []
        estres = []
        gradientes = []
        contador = 0 #reinicia el contador
        numero_guardado += 1
    
      

final = time.time()
delta_tiempo = (final-inicio)/60

print('El tiempo de ejecucion fue', delta_tiempo)


dat_out = Nombre_salida + '.dat' 
fix_out = Nombre_salida + '.fix'
geo_out = Nombre_salida + '.geo'

Borrar_file([dat_out,fix_out,geo_out])
