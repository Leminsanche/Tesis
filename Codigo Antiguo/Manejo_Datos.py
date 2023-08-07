
from launch.Funciones import *

Direccion_fom = 'Resultados/cubo_demiray/' 
Direccion_ROM  = 'SVD/cubo_demiray/'


nombre_variables_fom  = [ 'Desplazamientos_Demiray_cubo_',
                          'Gradientes_Demiray_cubo_',
                          'Tensiones_Demiray_cubo_']

variables_rom = ['Desplazmaientos', 'Gradientes','Tensiones' ]

for it, nombre_fom in enumerate(nombre_variables_fom): 

    info_FOM = [Direccion_fom,nombre_fom]

    info_ROM = [Direccion_ROM,variables_rom[it]]

    print('\n###########################  Guardando Datos de ',variables_rom[it],'  ###########################')
    Guardado_modos(info_FOM,info_ROM)