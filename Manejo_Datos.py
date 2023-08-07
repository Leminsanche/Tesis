
from launch.Funciones import *
from launch.ReduccionOrden import *
Direccion_fom = 'Resultados/Biaxial_Demiray/' 
Direccion_ROM  = 'SVD/Biaxial_Demiray/'


nombre_variables_fom  = ['Desplazamientos_Demiray_biaxial_',
                          'Gradientes_Demiray_biaxial_',
                         'Tensiones_Demiray_biaxial_']


                          
                          
                          

variables_rom = ['Desplazmaientos', 'Gradientes','Tensiones' ]
#print(nombre_variables_fom) 

for it, nombre_fom in enumerate(nombre_variables_fom):
    
    info_FOM = [Direccion_fom,nombre_fom]

    info_ROM = [Direccion_ROM,variables_rom[it]]

    print('\n###########################  Guardando Datos de ',variables_rom[it],'  ###########################')
    Guardado_modos(info_FOM,info_ROM)