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


nstep = 30
rangos_constantes = [[0.02,0.2,2],[0.3,1.5,3]]
constantes = Combinaciones(rangos_constantes)
#constantes = [[0.02,1.5]]
Ubicacion_caso = 'Casos/cubo/'
Nombre_msh = 'cubo.msh'
Nombre_dat = 'cubo_demiray'
Nombre_salida = 'bTest'
Nombre_info_salida = '_Demiray_cubo_'  ## Nombre caso EJ: _Tracion_Demiray_
Ubicacion_salida = 'Resultados/cubo_demiray/'

param = {'Cons1': 0.02,'Cons2': 1,'Penal': 20}  #{'Cons1': 0.02,'Cons2': 1,'Cons3': 1,'Penal': 20 }
#constantes = [[0.02,0.3],[0.02,1.5],[0.2,0.3],[0.2,1.5]]
#constantes = [[30e-3,3.57]]
file_msh = Ubicacion_caso + Nombre_msh
dat = Ubicacion_caso + Nombre_dat + '.dat' 
fix = Ubicacion_caso + Nombre_dat + '.fix'
geo = Ubicacion_caso + Nombre_dat + '.geo'




caso1 = VulcanHandler([dat,geo,fix],Nombre_salida)


lista_fail = []
contador = 0
numero_guardado = 0

for i in range(len(constantes)):

    print('######################## Simulacion ',i+1, 'de ',len(constantes),' ########################' )
    
    for jt,j in enumerate(param):
        if jt == (len(param) - 1):
            param[j] = constantes[i][0]*10

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


    if nstep == len(disp):
        contador = contador +1 #Cuenta las simulacion que se han realizado
        desplazamientos.append(disp)
        estres.append(stress)
        gradientes.append(gradientes_deformacion)
    
    else:
        lista_fail.append(param)

    
    if contador == 1:    #Guarda cuando se realizan n simulaciones (contador == n)
        nombre_T = 'Tensiones' + Nombre_info_salida + str(numero_guardado) + '.txt' 
        nombre_D = 'Desplazamientos' + Nombre_info_salida  + str(numero_guardado) + '.txt'
        nombre_G = 'Gradientes' + Nombre_info_salida  + str(numero_guardado) + '.txt'
    
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