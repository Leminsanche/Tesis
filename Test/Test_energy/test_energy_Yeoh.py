from launch.gradientes import Gradientes_nodales_Vulcan
from launch.vulcan_handler import VulcanHandler
from intervul.readpos import VulcanPosMesh
from launch.Funciones import *
import numpy as np
import matplotlib.pyplot as plt
import time
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from launch.Energias import *
import random

Ubicacion_caso = '/home/nicolas/Escritorio/launch_cases/Test/Test_energy/Cubo_Yeoh/'
Nombre_salida = 'Test_energy_yeoh'

a,b,c = 0.018412852746797336, 0.04638435127243134, 0.004321486477620494

param = {'Cons1': a,'Cons2': b,'Cons3': c,'Penal': 10000*a }



dat, geo, fix, file_msh = files_vulcan(Ubicacion_caso)

caso1 = VulcanHandler([dat,geo,fix],Nombre_salida)
caso1.run(param)
disp, stress = get_results(caso1.pathToPos())
fuerzas = Resultados_vulcan(caso1.pathToPos()).Fuerzas()
print('Numero de pasos simulados: ', len(disp))
gradientes_deformacion = Gradientes_nodales_Vulcan(file_msh,disp)
print('J global: ', gradientes_deformacion [1] )


dat_out = Nombre_salida + '.dat' 
fix_out = Nombre_salida + '.fix'
geo_out = Nombre_salida + '.geo'

Borrar_file([dat_out,fix_out,geo_out])

####################################### Fuerza Externa #######################################


#print(disp[-1,:,:])

mesh = pv.read(file_msh)
mesh.clear_data()

energia  =Energia_deformacion(gradientes_deformacion[0][:,-1]).Yeoh(a,b,c)

COO = mesh.points
energia_cell = []
#Volume_cell = []
for it, i in enumerate(mesh.cells_dict[12]):
    nodos = COO[i]
    energia_cell.append(np.mean(energia[i]))



#################################################### Volumen celdas ##################################################################

mesh.points +=  disp[-1,:,:]
#mesh.plot()

volumenes_cell  = []
for  i in mesh.compute_cell_sizes()["Volume"]:
    if  i != 0 :
        volumenes_cell.append(i)


energia_cell = np.array(energia_cell)#.reshape((-1,1))
volumenes_cell = np.array(volumenes_cell)#.reshape((1,-1))

print('Energía de Deformacion')
#print(volumenes_cell)
print(np.dot(volumenes_cell, energia_cell))
# Cambiar el volumen orgiinal al actual

########################################################## Energia Externa ##########################################################



Punto_extremo1 = [] # Extremo 1 es X
Punto_extremo2 = [] # Extremo 2 es Y

#print(mesh.points)
for it,i in  enumerate(mesh.points):
    if  i[-1] != 0:
        Punto_extremo1.append(it)
        

    elif i[-1] != 0 and  i[1] == 1.:
        Punto_extremo2.append(it)
        

#print(Punto_extremo1)




Fuerza_extremo1 = np.sum(fuerzas[:,Punto_extremo1,-1], axis = 1)
desplazamientos1 = disp[:,Punto_extremo1[0],-1]
#print(desplazamientos1)
#print(Fuerza_extremo1 )
############################################################### Area Bajo la curva ################################################



n = 3000
#Datos = [[desplazamientos1,Fuerza_extremo1],[desplazamientos2,Fuerza_extremo2] ]
Datos = [[desplazamientos1,Fuerza_extremo1]]

Landas = []
Fuerzas = []
for i in Datos:    
    Fuerza = i[1]
    landa = i[0]
    mini = min(landa)
    maxi = max(landa)
    
    X_n = np.linspace(mini,maxi,n)
    f = interpolate.interp1d(landa, Fuerza)
    Y_n = f(X_n)
    
    Landas.append(X_n)
    Fuerzas.append(Y_n)
    
    #plt.plot(landa,Fuerza)
#plt.grid()

print('Energía Externa')
print(Trapecio_discreto(Landas[0],Fuerzas[0]))

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches(10,5)
fig.suptitle('Fuerzas Extremos')
ax1.plot(desplazamientos1,Fuerza_extremo1)
ax1.set_title('Extremo X')
ax1.set_ylabel('Fuerza [N]')
ax1.set_xlabel('Desplazamiento [mm]')
ax1.fill_between(desplazamientos1,Fuerza_extremo1, 0, color='blue', alpha=.1)
ax1.grid()
#plt.show()
#ax2.plot(desplazamientos2,Fuerza_extremo2)
#ax2.set_title('Extremo Y')
#ax2.set_ylabel('Fuerza [N]')
#ax2.set_xlabel('Desplazamiento [mm]')
#ax2.grid()
#ax2.fill_between(desplazamientos2,Fuerza_extremo2, 0, color='blue', alpha=.1)