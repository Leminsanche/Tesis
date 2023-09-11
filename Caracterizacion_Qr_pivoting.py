from launch.Lanzadores import *
import numpy as np
import pyvista as pv
from launch.Funciones import *
from launch.Energias import *
import random
from launch.Funciones import *
from sklearn.metrics import mean_squared_error
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import trapz, simps


class MyProblem(ElementwiseProblem):

    def __init__(self,datos_y,Modelo,Gradiente,Volumenes,x_l, x_u,num_const):
        
        #self.datos_x = datos_x
        self.datos_y = datos_y
        self.Modelo = Modelo
        self.num_const = num_const
        self.Gradiente = Gradiente
        self.Volumenes  = Volumenes

        super().__init__(n_var=num_const,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl= x_l,
                         xu=x_u)

    def _evaluate(self, x, out, *args, **kwargs):
        aux = self.Modelo(x,self.Gradiente, self.Volumenes )
        f1 = abs(aux- self.datos_y )/ self.datos_y
        out["F"] = f1



def Energia_Ml_Demiray(C,Gradiente_recons,volumenes_cell):
    a , b  = C[0] , C[1]
    
    energia  = Energia_deformacion(Gradiente_recons).Demiray(a,b) 
    energia_cell =  np.mean(energia[mesh.cells_dict[12]],axis = 1)
    return np.dot(volumenes_cell, energia_cell)

def Energia_Ml_Yeoh(C,Gradiente_recons):
    a , b, c  = C[0] , C[1], C[2]
    E = Energia_deformacion(Gradiente_recons).Yeoh(a,b,c)
    return np.array([sum(E)])

def Energia_Ml_Mooney(C,Gradiente_recons):
    a , b, c  = C[0] , C[1], C[2]
    E = Energia_deformacion(Gradiente_recons).Mooney(a,b,c)
    return np.array([sum(E)])



























if __name__ == '__main__':
    ########################################################## Simulacion Realizada ##########################################################
    a,b,c = random.uniform(0.1,0.9) , random.uniform(0,0) , random.uniform(3,14)
    a,c = 0.27472632, 8.94008299
    #a,b,c = random.uniform(0.001,0.03) , random.uniform(0.002,0.08) , random.uniform(0.002,0.01)
    desplazamientos, tensiones, gradientes, malla, fuerzas = Vulcan().Biaxial_Demiray(a,c,penal = 10000, fuerzas_flag = True)
    x,y,z = desplazamientos.shape
    D = desplazamientos.transpose((1,2,0)).reshape((y*z,x)) 


    mesh = pv.read(malla)
    mesh.clear_data()
    mesh_f = mesh.copy()

    ########################################################## Energia interna ##########################################################

    energia  =Energia_deformacion(gradientes[0][:,-1]).Demiray(a,c)


    #################################################### Volumen celdas ##################################################################

    mesh_f.points +=  ManejoDatos(D[:,-1],3).ModoVector()
    #mesh.plot()

    energia_cell = np.mean(energia[mesh.cells_dict[12]],axis = 1)
    volumenes_cell  = mesh_f.compute_cell_sizes()["Volume"][np.nonzero(mesh_f.compute_cell_sizes()["Volume"])]


    energia_cell = np.array(energia_cell)#.reshape((-1,1))
    volumenes_cell = np.array(volumenes_cell)#.reshape((1,-1))

    print('Energía de Deformacion')
    print(np.dot(volumenes_cell, energia_cell))

    ########################################################## Energia Externa ##########################################################
    Punto_extremo1 = [] # Extremo 1 es X
    Punto_extremo2 = [] # Extremo 2 es Y
    aux_puntos_sel = []

    for it,i in  enumerate(mesh.points):
        if   i[0] == 20:
            Punto_extremo1.append(it)
            aux_puntos_sel.append(1)

        elif  i[1] == 20:
            Punto_extremo2.append(it)
            aux_puntos_sel.append(1)

        else:
            aux_puntos_sel.append(0)

    Fuerza_extremo1 = np.sum(fuerzas[:,Punto_extremo1,0], axis = 1)
    desplazamientos1 = desplazamientos[:,Punto_extremo1[int(len(Punto_extremo1)/2-1)],0]
    Fuerza_extremo2 = np.sum(fuerzas[:,Punto_extremo2,1],axis = 1)
    desplazamientos2 = desplazamientos[:,Punto_extremo2[int(len(Punto_extremo2)/2-1)],1]

    ############################################################### Area Bajo la curva ################################################
    Energia_x = simps(Fuerza_extremo1,desplazamientos1)
    Energia_y = simps(Fuerza_extremo2,desplazamientos2)
    Energia_Externa = Energia_x + Energia_y
    # n = 3000
    # Datos = [[desplazamientos1,Fuerza_extremo1],[desplazamientos2,Fuerza_extremo2] ]
    # #Datos = [[desplazamientos1,Fuerza_extremo1]]

    # Landas = []
    # Fuerzas = []
    # for i in Datos:    
    #     Fuerza = i[1]
    #     landa = i[0]
    #     mini = min(landa)
    #     maxi = max(landa)
        
    #     X_n = np.linspace(mini,maxi,n)
    #     f = interpolate.interp1d(landa, Fuerza)
    #     Y_n = f(X_n)
        
    #     Landas.append(X_n)
    #     Fuerzas.append(Y_n)
        
    #     #plt.plot(landa,Fuerza)
    # #plt.grid()

    print('Energía Externa')
    #Energia_Externa = Trapecio_discreto(Landas[0],Fuerzas[0])*2
    print(Energia_Externa)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10,5)
    fig.suptitle('Fuerzas Extremos')
    ax1.plot(desplazamientos1,Fuerza_extremo1)
    ax1.set_title('Extremo X')
    ax1.set_ylabel('Fuerza [N]')
    ax1.set_xlabel('Desplazamiento [mm]')
    ax1.fill_between(desplazamientos1,Fuerza_extremo1, 0, color='blue', alpha=.1)
    ax1.grid()
    ax2.plot(desplazamientos2,Fuerza_extremo2)
    ax2.set_title('Extremo Y')
    ax2.set_ylabel('Fuerza [N]')
    ax2.set_xlabel('Desplazamiento [mm]')
    ax2.grid()
    ax2.fill_between(desplazamientos2,Fuerza_extremo2, 0, color='blue', alpha=.1)
    #plt.show()


    ##################################### Seleccion Qr Pivoting ######################################################
    C = np.loadtxt('SVD/Biaxial2/Matriz_C.txt')
    indices_bases = np.loadtxt('SVD/Biaxial2/indices_base.txt')
    print(C.shape)
    desplazamientos_samples = D[:,-1][indices_bases.astype(int)]

    # Se cargan los Modos de los desplazamientos conocidos
    Modos_desp = np.loadtxt('SVD/Biaxial2/Modos_Desplazmaientos.txt')[:,:8]
    A = np.matmul(C,Modos_desp)
    coef = np.linalg.solve(A,np.array(desplazamientos_samples))

    ROM_u  = np.matmul(Modos_desp, coef)
    ROM_F  = Gradientes_nodales_Vulcan(malla,np.array([ManejoDatos(np.matmul(Modos_desp, coef),3).ModoVector()]))[0]


    mse_desp = mean_squared_error(D[:,-1], ROM_u)
    mse_grad = mean_squared_error(gradientes[0][:,-1], ROM_F)
    print('Mean Square Error Desplazamientos: ',mse_desp)
    print('Mean Square Error Gradientes: ',mse_grad)


#################################################### Caracterizacion ######################################################################

problem = MyProblem(Energia_Externa,Energia_Ml_Demiray,ROM_F,volumenes_cell,[0.1,3],[0.9,14],2)




## algoritmo genetico ##
algorithm = GA(
    pop_size=10,
    eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               verbose=True)

print("Best solution found in GA: \nX = %s\nF = %s" % (res.X, res.F))
coef_GA  = res.X

# algorithm = CMAES(x0=np.array((random.uniform(0.1,0.9) , random.uniform(3,14))))
algorithm = CMAES(x0= coef_GA)


res = minimize(problem,
               algorithm,
               seed=111,
               verbose=True)

print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
coef_CMAES  = res.X



print('################################### Constantes ################################### ')


print('Coeficientes Originales')
print(a,c)
print('\n')
print('Coeficientes GA')
print(coef_GA)
print('\n')
print('Coeficientes GMAES')
print(coef_CMAES)

print('################################### Energia ################################### ')

print('Energía Externa')
print(Energia_Externa)
print('\n')
print('Energía GA')
energia_recons = Energia_deformacion(ROM_F).Demiray(coef_GA [0],coef_GA[1])  
energia_cell_recons =  np.mean(energia_recons[mesh.cells_dict[12]],axis = 1)
print(np.dot(volumenes_cell, energia_cell_recons))
print('Energía CMAES')
energia_recons = Energia_deformacion(ROM_F).Demiray(coef_CMAES[0],coef_CMAES[1])  
energia_cell_recons =  np.mean(energia_recons[mesh.cells_dict[12]],axis = 1)
print(np.dot(volumenes_cell, energia_cell_recons))


