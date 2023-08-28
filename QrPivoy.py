import numpy as np
from scipy import linalg

Direccion_file = 'SVD/cubo_demiray/Modos_Desplazmaientos.txt'
Direccion_Salida = 'SVD/cubo_demiray/'
Nombre_Salida = 'Matriz_C'


A = np.loadtxt(Direccion_file)
pivot = linalg.qr(A,pivoting = True)[2]
C = np.zeros((A.shape[1],A.shape[0]))
for it,i in enumerate(pivot):
    C[it,i] =  1

texto = Direccion_Salida + Nombre_Salida + '.npz'
np.savez_compressed(texto,C)