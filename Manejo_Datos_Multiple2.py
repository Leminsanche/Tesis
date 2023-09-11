
from launch.Funciones import *
from launch.ReduccionOrden import *
import numpy as np
import matplotlib.pyplot as plt

Direcciones_fom = ['Resultados/Biaxial_Demiray_20/','Resultados/Biaxial_Yeoh_20/'] 
Direccion_ROM  = 'SVD/Biaxial4/'

nombres_variables_fom  = [[ 'Desplazamientos_Demiray20_Biaxial_',
                          'Gradientes_Demiray20_Biaxial_',
                          'Tensiones_Demiray20_Biaxial_'],

                          ['Desplazamientos_Yeoh20_Biaxial_',
                          'Gradientes_Yeoh20_Biaxial_',
                          'Tensiones_Yeoh20_Biaxial_']
]

#variables_rom = ['Desplazmaientos', 'Gradientes','Tensiones' ]
variables_rom = ['Desplazmaientos']

infos_FOM = []
for it,dir_fom in enumerate(Direcciones_fom):
    for it2, nom_var_fom in enumerate(nombres_variables_fom[it]):
        info_FOM = [dir_fom,nom_var_fom]
        infos_FOM.append(info_FOM)




Datos_unir = []
for it, var_fom in enumerate(variables_rom):
    aux = [var_fom]
    for i in range(int(len(infos_FOM)/len(variables_rom))):
        aux.append(infos_FOM[it -(len(variables_rom))*i])
    Datos_unir.append(aux)

for it, info_FOM in enumerate(Datos_unir):
    print('\n Variables', info_FOM[0] )
    Variable_str = info_FOM[0]
    MS = Armado_MSnaptchots(info_FOM[1],Variable_str)#Matriz Snaptchot

    for i in info_FOM[2:]:
        MS = np.concatenate((MS,Armado_MSnaptchots(i,Variable_str)),axis =1)
        
        #print('media')
        #print(Mean.shape)
        #des = np.std(MS, axis = 1)

        #for kt,k in enumerate(des):
        #    if k ==0:
        #        des[kt] = 1

        #print('desviacion')
        #print(des.shape)
    Mean = np.mean(MS,axis = 1)
    for i in range(len(MS[0,:])):
        MS[:,i] = MS[:,i] - Mean
        #MS[:,i] = MS[:,i] / des

    nombre_media = Direccion_ROM + 'Mean_' +Variable_str + '.txt'
    np.savetxt(nombre_media,Mean)
    print(MS.shape)
    SVD_Modos_dask(MS,[Direccion_ROM,Variable_str],Grafico_r_variable = False, r = 8)

    del MS












