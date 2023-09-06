import numpy as np
import pyvista as pv
import collections

def Gradientes_nodales(COO_INI, COO_FIN, CONECTIVIDAD, list_for_f):
    """
    Funcion que calcula el tensor gradiente de deformacion (F) para cada nodo
    Los nodos que coinciden con mas de un elemento se realiza un promedio del tensor F
    Esta funcion retorna los gradientes de deformacion para cada nodo en forma de un array columna
    
    input:
    COO_ini = Coord_get_nodesenadas iniciales del problemas
    COO_fin = Coordenadas finales del problemas
    conectividasd = conectividad de los elementos de la malla
    """
    
    COO_iniciales = COO_INI
    COO_finales = COO_FIN
    conectividad = CONECTIVIDAD


    elementos = Hexs(COO_iniciales,CONECTIVIDAD)
    F = elementos.f(COO_finales)
    F_nodes = np.zeros((len(COO_iniciales), 3, 3))

    for i, ilist in enumerate(list_for_f):
        iilist = np.array(ilist, dtype=int)
        F_nodes[i] = F[iilist[:,0], iilist[:,1], :, :].mean(axis=-3)
        
    #print(F_nodes.shape)
    #print(COO_INI.shape)
    #aux = np.einsum('bji,bj->bi', F_nodes, COO_INI)
    #print(aux == COO_FIN)
    #print(aux)
    #print(COO_FIN)
    return F_nodes


def Gradientes_nodales_Vulcan(file,disp):
    """
    Funcion realizada para calcular los gradientes para cada step de carga
    funcion pensada para funcionar acoplada a vulcan handler
    input
    
    file = direccion en la cual se encuentra el archivo .msh
    desp = Matriz de 3 dimensiones con la informacion (step x nodos x dimension)

    output: Una matriz con los gradientes para cada step ordenadsos como vector columna para cada step
    
    """



    mesh = pv.read(file)
    mesh.clear_data()
    COO = mesh.points
    #print(COO)
    desplazamientos = disp
    num_steps , num_nodos, num_dim = disp.shape
    COO_def = [] ## Coordenadas deformadas

    for i in range(num_steps):
        desp = desplazamientos[i]
        COO_n = COO + desp 
        COO_def.append(COO_n)
        
    ###########################################

    #for  it,i in enumerate(COO_def):
    #    print('step', it ,'\n', i)
    ###########################################

    nodes_repeated = {}
    for i in range(len(COO)):
        nodes_repeated[i] = []

    for i, ielem in enumerate(mesh.cells_dict[12]):
        for it, nodo in enumerate(ielem):
            nodes_repeated[nodo].append([i, it])

    ordered_nodes_repeated = collections.OrderedDict(sorted(nodes_repeated.items()))
    nodes_of_elements = np.array(list(ordered_nodes_repeated.keys()))
    list_for_f = np.array(list(ordered_nodes_repeated.values()), dtype=object)

    Gradientes_step = np.zeros((len(COO), 3, 3, len(COO_def)))
    # Gradientes_step = np.zeros((len(COO)*9,len(COO_def)))

    determinantes = [] ################################## FLAG

    for istep, iCOO_def  in enumerate(COO_def):
        #print('########################################## Paso, istep ##########################################')
        Gradientes_step[:,:,:,istep] = Gradientes_nodales(COO,iCOO_def, mesh.cells_dict[12], list_for_f)
        determinantes.append(np.linalg.det(Gradientes_step[:,:,:,istep])) ########################################### FLAG

    #print('Promedio J global = ',np.mean(np.array(determinantes)))  ################################ FLAG
    J_global  = np.mean(np.array(determinantes))
    Gradientes_step = Gradientes_step.reshape(((len(COO)*9,len(COO_def))))
        
    return Gradientes_step, J_global



class Hex:
    def __init__(self, nodes, conn):
        self.conn = conn
        self.nodes = nodes[conn]
        self.nnodes = 8

    def _get_nodes(self, x):
        return x[self.conn,:]


    def N_func(self, xi):
        xi0 = xi[0]
        xi1 = xi[1]
        xi2 = xi[2]
        N1 = (1.0 - xi0)*(1.0 - xi1)*(1.0 - xi2)/8.0  
        N2 = (1.0 + xi0)*(1.0 - xi1)*(1.0 - xi2)/8.0  
        N3 = (1.0 + xi0)*(1.0 + xi1)*(1.0 - xi2)/8.0  
        N4 = (1.0 - xi0)*(1.0 + xi1)*(1.0 - xi2)/8.0  
        N5 = (1.0 - xi0)*(1.0 - xi1)*(1.0 + xi2)/8.0  
        N6 = (1.0 + xi0)*(1.0 - xi1)*(1.0 + xi2)/8.0  
        N7 = (1.0 + xi0)*(1.0 + xi1)*(1.0 + xi2)/8.0
        N8 = (1.0 - xi0)*(1.0 + xi1)*(1.0 + xi2)/8.0
        return np.array([N1, N2, N3, N4, N5, N6, N7, N8])

    def der_N_fun(self, xi):
        xi0 = xi[0]
        xi1 = xi[1]
        xi2 = xi[2]
        return np.array([[  -(1.0 - xi1)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 - xi1)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 + xi1)*(1.0 - xi2)/8.0,  (1.0 + xi0)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 + xi1)*(1.0 - xi2)/8.0,  (1.0 - xi0)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 - xi1)*(1.0 + xi2)/8.0, -(1.0 - xi0)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 - xi1)*(1.0 + xi2)/8.0, -(1.0 + xi0)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 + xi1)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 + xi1)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 + xi1)/8.0],
                         ]) 

    def der_X_xi(self, xi):  # 7.6b
        return np.einsum('ai,aj', self.nodes, self.der_N_fun(xi))

    def der_N_X(self, xi):  # 7.6b
        inv_der_X_xi = np.linalg.inv(self.der_X_xi(xi).T)
        out = np.matmul(inv_der_X_xi,self.der_N_fun(xi).T).T
        ##print(out.shape)
        return out

    def der_x_xi(self, x, xi):  # 7.11a
        return np.einsum('ai,aj', x, self.der_N_fun(xi))

    def der_N_x(self, x, xi):  # 7.11b
        inv_der_x_xi = np.linalg.inv(self.der_x_xi(x, xi).T)
        
        return np.matmul(inv_der_x_xi,self.der_N_fun(xi).T).T

    def f(self, x_n):  # gradiente de deformacion -- 7.5
        
        x = self._get_nodes(x_n)
        Fs = []
        
        puntos_iso = np.array([[-1,-1,-1],
                               [ 1,-1,-1],
                               [ 1, 1,-1],
                               [-1, 1,-1],
                               [-1,-1, 1],
                               [ 1,-1, 1],
                               [ 1, 1, 1],
                               [-1, 1, 1] ])
        
        #print('Nodos')
        #print(x)
        #print('Gradientes F \n')
        
        #print('Gradiente de Deformacion')
        for pi in puntos_iso:
            xi = pi
            F = np.einsum('ai,aj->ij', x, self.der_N_X(xi))

            
            #print(F)
            Fs.append(F)
        


        
        return Fs
        
        
        
class Hexs:
    def __init__(self, nodes, conn):
        self.conn = conn
        self.nodes = nodes[conn]
        self.nnodes = 8


        puntos_iso = np.array([[-1,-1,-1],
                               [ 1,-1,-1],
                               [ 1, 1,-1],
                               [-1, 1,-1],
                               [-1,-1, 1],
                               [ 1,-1, 1],
                               [ 1, 1, 1],
                               [-1, 1, 1] ])

        self.der_N_X_esquinas = [self.der_N_X(i) for i in puntos_iso]
        self.der_N_X_esquinas = np.array(self.der_N_X_esquinas).transpose((1,0,2,3))

    def _get_nodes(self, x):
        return x[self.conn,:]


    def N_func(self, xi):
        xi0 = xi[0]
        xi1 = xi[1]
        xi2 = xi[2]
        N1 = (1.0 - xi0)*(1.0 - xi1)*(1.0 - xi2)/8.0  
        N2 = (1.0 + xi0)*(1.0 - xi1)*(1.0 - xi2)/8.0  
        N3 = (1.0 + xi0)*(1.0 + xi1)*(1.0 - xi2)/8.0  
        N4 = (1.0 - xi0)*(1.0 + xi1)*(1.0 - xi2)/8.0  
        N5 = (1.0 - xi0)*(1.0 - xi1)*(1.0 + xi2)/8.0  
        N6 = (1.0 + xi0)*(1.0 - xi1)*(1.0 + xi2)/8.0  
        N7 = (1.0 + xi0)*(1.0 + xi1)*(1.0 + xi2)/8.0
        N8 = (1.0 - xi0)*(1.0 + xi1)*(1.0 + xi2)/8.0
        return np.array([N1, N2, N3, N4, N5, N6, N7, N8])

    def der_N_fun(self, xi):
        xi0 = xi[0]
        xi1 = xi[1]
        xi2 = xi[2]
        return np.array([[  -(1.0 - xi1)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 - xi1)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 + xi1)*(1.0 - xi2)/8.0,  (1.0 + xi0)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 + xi1)*(1.0 - xi2)/8.0,  (1.0 - xi0)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 - xi1)*(1.0 + xi2)/8.0, -(1.0 - xi0)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 - xi1)*(1.0 + xi2)/8.0, -(1.0 + xi0)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 + xi1)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 + xi1)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 + xi1)/8.0],
                         ]) 

    def der_X_xi(self, xi):  # 7.6b
        return np.einsum('...ai,aj', self.nodes, self.der_N_fun(xi))

    def der_N_X(self, xi):  # 7.6b
        temp = self.der_X_xi(xi).transpose(0,2,1)
        inv_der_X_xi = np.linalg.inv(temp)
        out = np.matmul(inv_der_X_xi, self.der_N_fun(xi).T).transpose(0,2,1)
        ##print(out.shape)
        return out

    def der_x_xi(self, x, xi):  # 7.11a
        return np.einsum('ai,aj', x, self.der_N_fun(xi))

    def der_N_x(self, x, xi):  # 7.11b
        temp = self.der_x_xi(x, xi).transpose(0,2,1)
        inv_der_x_xi = np.linalg.inv(temp)
        
        return np.matmul(inv_der_x_xi,self.der_N_fun(xi).T).transpose(0,2,1)

    def f(self, x_n):  # gradiente de deformacion -- 7.5
        #print("disp", x_n)
        x = self._get_nodes(x_n)
        Fs = []
        
        F = np.einsum('eai,exaj->exij', x, self.der_N_X_esquinas)
    
        #print("test", F)
        return F
    


class energy(Hexs):

    def __init__(self, nodes, conn):
        self.conn = conn
        self.nodes = nodes[conn]
        self.nnodes = 8

        #print(f'conn: {conn}')

        puntos_iso = np.array([[-1,-1,-1],
                               [ 1,-1,-1],
                               [ 1, 1,-1],
                               [-1, 1,-1],
                               [-1,-1, 1],
                               [ 1,-1, 1],
                               [ 1, 1, 1],
                               [-1, 1, 1] ])
        

        self.gauss_points = np.array([[-1/3**0.5, -1/3**0.5, -1/3**0.5],
                                      [ 1/3**0.5, -1/3**0.5, -1/3**0.5],
                                      [ 1/3**0.5,  1/3**0.5, -1/3**0.5],
                                      [-1/3**0.5,  1/3**0.5, -1/3**0.5],
                                      [-1/3**0.5, -1/3**0.5,  1/3**0.5],
                                      [ 1/3**0.5, -1/3**0.5,  1/3**0.5],
                                      [ 1/3**0.5,  1/3**0.5,  1/3**0.5],
                                      [-1/3**0.5,  1/3**0.5,  1/3**0.5],
                                      ])
        
        self.der_N_X_gp = [self.der_N_X(i) for i in self.gauss_points]

        self.der_N_X_gp = np.array(self.der_N_X_gp).transpose((1,0,2,3))

    def Delphino(self,F,c1,c2):
        F_T = np.moveaxis(F,-1,-2)
        C = np.matmul(F_T,F)
        I1,I2,I3 = np.einsum('...ii',C), 0.5 * (np.einsum('...ii',C)**2 - np.einsum('...ii',np.matmul(C,C))), np.linalg.det(C)
        aux = I3**(-1/3)
        energia  =  (c1/c2) * (np.e**(c2*0.5*(I1*aux-3)) - 1 )    
        #print(energia)    
        return energia
    
    def f_gauss(self, x_n):  # gradiente de deformacion -- 7.5
        #print("disp", x_n)
        x = self._get_nodes(x_n)
        #print(f"xn: {x}")
        #print(self.der_N_X_gp)
        F = np.einsum('eai,exaj->exij', x, self.der_N_X_gp)
        #print("test", F)
        return F


    def Delphino_E(self,x_n,c1,c2):
        # x_n nuevas coordenadas
        #print(x_n)
        x = self._get_nodes(x_n)
        F = self.f_gauss(x_n)
        # print(F.shape)
        temp = self.Delphino(F,c1,c2)
        # print(temp.shape)
        micro = []
        for it, gp in enumerate(self.gauss_points):
            aux = np.linalg.det(self.der_x_xi(x, gp))
            micro.append(aux)
        e_t = np.dot(temp,np.array(micro))
        # print(np.array(micro).sum())
        # # print(e_t)
        # #print(micro)
        # print(x_n)
        return e_t