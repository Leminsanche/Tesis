import os
import numpy as np
import matplotlib.pyplot as plt
import time
import dask.array as da

def Armado_MSnaptchots(INFO_FOM,Variable_str,n=3):
    """
    Funcion que realiza el armado de la matriz de snaptchots.
    input
    INFO_FOM (type list)  -> Direccion_FOM: Direccion de la carpeta donde estan los resultados FOM. type: str 
                             nombre_FOM:  Nombre de los archivos que contienen la informacion FOM. type: str

    n = numero de variables guardadas del FOM. type: int
    """
    Direccion_FOM,nombre_FOM  = INFO_FOM   

    direccion_resultados = Direccion_FOM
    nombre_variable =  nombre_FOM

    lst = os.listdir(direccion_resultados)
    number_files = len(lst)
    print('Archivos Totales: ',number_files)

    numero_txt = number_files/n -1
    print('Numero de txt con la variable ', Variable_str ,int(numero_txt))

    Datos_V = []

    inicio0 = time.time()

    for i in range(int(numero_txt)+1):
        #print(nombre_variable)
        #nombre_V = direccion_resultados + nombre_variable + str(i) + '.txt' 
        nombre_V = direccion_resultados + nombre_variable  + str(i) + '.npz' 
        Vaux = np.load(nombre_V) #V = np.loadtxt(nombre_V)
        V = Vaux['arr_0']
        Datos_V.append(V)
    #print(len(Datos_V))
    Variable = Datos_V[0]
    for i in Datos_V[1:]:
        Variable = np.concatenate((Variable,i),axis =1)
    fin0 = time.time()

    print('Matriz de snaptchot armada en: ', (fin0 - inicio0)/60, '[min]')   
    #print('Dimens}iones Matriz de snaptchots', Variable.shape)

    del Datos_V # Se borra la lista de informacion para liberar memoria 

    return Variable


def SVD_Modos(Variable,INFO_ROM,Grafico_r_variable = False,r = 8):
    """
    Funcion para realizar la seleccion de modos del SVD

    Grafico_r_variable: Valor que indica si se quiere ver el grafico del SVD para revisar el comportamiento del grafico. type: Boolean 
    
    r: numero de modos que se consideraran para guardar los resultados. type: int
    """

    Direccion_SVD, Variable_str = INFO_ROM
    #print(Variable_str)

    nombre_variable_modos = 'Modos_' + Variable_str +'.txt'
    nombre_coef_V = 'Coeficientes_' + Variable_str +'.txt' 

    inicio0 = time.time()
    Uv,Sv,Vv = np.linalg.svd(Variable,full_matrices=False)
    fin0 = time.time()

    print('SVD Realizado en: ', (fin0 - inicio0)/60, '[min]') 

    AcumV = []
    for it,i in enumerate(Sv):
        aux = sum(Sv[:it])
        AcumV.append(aux)

######################### Graficos #################################
    if Grafico_r_variable == True: 

        plt.plot(AcumV/sum(Sv),'o')
        plt.title(Variable_str)
        plt.ylabel('Energia')
        plt.xlabel('modos')
        plt.grid()
        plt.show()

        r = int(input('ingrese numero de snaptchots: '))

######################### Graficos #################################

    B_V = Uv[:,:r]
    print('Energía acumulada en ',Variable_str,'es de: ',(AcumV[r]/sum(Sv))*100,'%')
    nombre_pesos = 'Pesos_' +Variable_str + '.txt'
    np.savetxt(Direccion_SVD + nombre_pesos  ,Sv[:100])
    #### Guardado de datos y modos mas improtantes
    np.savetxt(Direccion_SVD + nombre_variable_modos ,B_V)

    ## Caluclo de Coeficientes reduccion de orden X = fhi A

    Sv_red = np.zeros((r,r))
    Vv_red = Vv[:r,:]

    ###### Para tensiones
    it = 0
    for i in range(r):
        for j in range(r):
            if i == j:
                Sv_red[i,j] = Sv[it]
                it = it +1
                
    A_V = np.matmul(Sv_red,Vv_red)
    np.savetxt(Direccion_SVD + nombre_coef_V,A_V)
    print('Archivos escritos en',Direccion_SVD,'\n')

    return 



def Guardado_modos(INFO_FOM,INFO_ROM,Grafico_r_variable = False, r = 8, n = 3):
    """
    Funcion que guarda los coeficientes del SVD y los modos principales
    input
    INFO_FOM (type list)  -> Direccion_FOM: Direccion de la carpeta donde estan los resultados FOM. type: str 
                             nombre_FOM:  Nombre de los archivos que contienen la inbformacion FOM. type: str

    INFO_ROM (type list)  -> Direccion_SVD: Direccion donde se guardaran los txt con los modos principales y Coeficientes conocidos. type: str
                             Variable_str: Nombre de la variable que se guardaran los resultados
    
    
    Grafico_r_variable: Valor que indica si se quiere ver el grafico del SVD para revisar el comportamiento del grafico. type: Boolean 
    
    r: numero de modos que se consideraran para guardar los resultados. type: int
    n = numero de variables guardadas del FOM. type: int

    output
    Mensaje de confirmacion
    """
    Direccion_FOM, nombre_FOM  = INFO_FOM
    Direccion_SVD, Variable_str = INFO_ROM


    direccion_resultados = Direccion_FOM
    direccion_svd = Direccion_SVD
    nombre_variable =  nombre_FOM

    nombre_variable_modos = 'Modos_' + Variable_str +'.txt'
    nombre_coef_V = 'Coeficientes_' + Variable_str +'.txt' 

    Variable = Armado_MSnaptchots(INFO_FOM,Variable_str,n)
    print(Variable.shape)
    inicio0 = time.time()
    Uv,Sv,Vv = np.linalg.svd(Variable,full_matrices=False)
    fin0 = time.time()

    print('SVD Realizado en: ', (fin0 - inicio0)/60, '[min]') 
    AcumV = []

    for it,i in enumerate(Sv):
        aux = sum(Sv[:it])
        AcumV.append(aux)
            

######################### Graficos #################################
    if Grafico_r_variable == True: 

        plt.plot(AcumV/sum(Sv),'o')
        plt.title(nombre_variable)
        plt.ylabel('Energia')
        plt.xlabel('modos')
        plt.grid()
        plt.show()

        r = int(input('ingrese numero de snaptchots: '))

######################### Graficos #################################


    B_V = Uv[:,:r]
    print('Energía acumulada en ',Variable_str,'es de: ',(AcumV[r]/sum(Sv))*100,'%')
    nombre_pesos = 'Pesos_' +Variable_str + '.txt'
    np.savetxt(direccion_svd + nombre_pesos ,Sv[:100])
    #### Guardado de datos y modos mas improtantes
    np.savetxt(direccion_svd + nombre_variable_modos ,B_V)

    ## Caluclo de Coeficientes reduccion de orden X = fhi A

    Sv_red = np.zeros((r,r))
    Vv_red = Vv[:r,:]

    ###### Para tensiones
    it = 0
    for i in range(r):
        for j in range(r):
            if i == j:
                Sv_red[i,j] = Sv[it]
                it = it +1
                
    A_V = np.matmul(Sv_red,Vv_red)
    np.savetxt(direccion_svd + nombre_coef_V,A_V)
    print('Archivos escritos en',direccion_svd)

    return


def SVD_Modos_dask(Variable,INFO_ROM,Grafico_r_variable = False,r = 25):
    """
    Funcion para realizar la seleccion de modos del SVD

    Grafico_r_variable: Valor que indica si se quiere ver el grafico del SVD para revisar el comportamiento del grafico. type: Boolean 
    
    r: numero de modos que se consideraran para guardar los resultados. type: int
    """

    Direccion_SVD, Variable_str = INFO_ROM
    #print(Variable_str)

    nombre_variable_modos = 'Modos_' + Variable_str +'.txt'
    nombre_coef_V = 'Coeficientes_' + Variable_str +'.txt' 

    inicio0 = time.time()
    #Uv,Sv,Vv = np.linalg.svd(Variable,full_matrices=False)
    Uv,Sv,Vv = da.linalg.svd_compressed(da.array(Variable),150)
    Uv,Sv,Vv = np.array(Uv) , np.array(Sv) , np.array(Vv) 
    fin0 = time.time()

    print('SVD Realizado en: ', (fin0 - inicio0)/60, '[min]') 

    AcumV = []
    for it,i in enumerate(Sv):
        aux = sum(Sv[:it])
        AcumV.append(aux)



    B_V = Uv[:,:r]
    print('Energía acumulada en ',Variable_str,'es de: ',(AcumV[r]/sum(Sv))*100,'%')
    nombre_pesos = 'Pesos_' +Variable_str + '.txt'
    np.savetxt(Direccion_SVD + nombre_pesos  ,Sv[:100])
    #### Guardado de datos y modos mas improtantes
    np.savetxt(Direccion_SVD + nombre_variable_modos ,B_V)

    ## Caluclo de Coeficientes reduccion de orden X = fhi A

    Sv_red = np.zeros((r,r))
    Vv_red = Vv[:r,:]

    ###### Para tensiones
    it = 0
    for i in range(r):
        for j in range(r):
            if i == j:
                Sv_red[i,j] = Sv[it]
                it = it +1
                
    A_V = np.matmul(Sv_red,Vv_red)
    np.savetxt(Direccion_SVD + nombre_coef_V,A_V)
    print('Archivos escritos en',Direccion_SVD,'\n')

    return 