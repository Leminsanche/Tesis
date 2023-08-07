from launch.vulcan_handler import VulcanHandler
from intervul.readpos import VulcanPosMesh
from launch.Funciones import *
import numpy as np
import matplotlib.pyplot as plt
import time
import pyvista as pvsss

inicio = time.time()


desplazamientos = []
estres = []
gradientes = []

constantes = []
constantes1 = np.linspace(0.02,0.2,18)
constantes2 = np.linspace(0.3,1.5,12)

for i in constantes1:
    for j in constantes2:
        constantes.append([i,j])
        
Ubicacion_caso = 'Casos/cubo/'
Nombre_msh = 'cubo2.msh'
Nombre_dat = 'cubo_demiray'
Nombre_salida = 'bTest'
Nombre_info_salida = '_Demiray_cubo_'  ## Nombre caso EJ: _Tracion_Demiray_
Ubicacion_salida = 'Resultados/cubo_demiray/'


file_msh = Ubicacion_caso + Nombre_msh
dat = Ubicacion_caso + Nombre_dat + '.dat' 
fix = Ubicacion_caso + Nombre_dat + '.fix'
geo = Ubicacion_caso + Nombre_dat + '.geo'

#constantes = [[0.02,0.3],[0.02,1.5],[0.2,0.3],[0.2,1.5]]
constantes = [[30e-3,3.77]]
param = {'Cons1': 0.02,'Cons2': 1,'Penal': 20 }
caso1 = VulcanHandler([dat,geo,fix],Nombre_salida)

contador = 0
numero_guardado = 0

for i in range(len(constantes)):

    print('######################## Simulacion ',i+1, 'de ',len(constantes),' ########################' )
    
    CONST1,CONST2 =  constantes[i]
    
    param['Cons1'] = CONST1
    param['Cons2'] = CONST2
    param['Penal'] = CONST1*1000
    
    print('Constante 1: ',CONST1,'\nConstante 2: ',CONST2,'\nPenalizador: ',CONST1*1000)
    
    caso1.run(param)
    disp, stress = get_results(caso1.pathToPos())
    gradientes_deformacion = Gradientes_nodales_Vulcan(file_msh,disp)

    print('Simulacion Finalizada\n')
    desplazamientos.append(disp)
    estres.append(stress)
    gradientes.append(gradientes_deformacion)

    
    #for  i in ManejoDatos(gradientes[0][:,-1],9).ModoVector():
    #    print(i.reshape(3,3))
    
    if contador == 0:    
        nombre_T = 'Tensiones' + Nombre_info_salida + str(numero_guardado) + '.txt' 
        nombre_D = 'Desplazamientos' + Nombre_info_salida  + str(numero_guardado) + '.txt'
        nombre_G = 'Gradientes' + Nombre_info_salida  + str(numero_guardado) + '.txt'
    
        save_result(nombre_T ,nombre_D,nombre_G,Ubicacion_salida,desplazamientos,estres,gradientes)
        
        desplazamientos = []
        estres = []
        gradientes = []
        contador = 0
        numero_guardado += 1
    contador = contador +1
      
    
final = time.time()
delta_tiempo = (final-inicio)/60

print('El tiempo de ejecucion fue', delta_tiempo)


dat_out = Nombre_salida + '.dat' 
fix_out = Nombre_salida + '.fix'
geo_out = Nombre_salida + '.geo'

Borrar_file([dat_out,fix_out,geo_out])


