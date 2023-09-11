from launch.Funciones import *
class Energia_deformacion():

    def __init__(self,gradiente):

        self.gradiente = gradiente.reshape((int(len(gradiente)/9),3,3))
        #determinantes = np.linalg.det(self.gradiente)


    def Cauchy_green_left(self):
        
        grad_T  = np.moveaxis(self.gradiente,-1,-2) 
        C = np.matmul(grad_T,self.gradiente)
        J = np.power(np.linalg.det(self.gradiente), (-2/3))
        #print(J)
        C_iso  = C[:] * J[:,np.newaxis,np.newaxis] 
        return C_iso
    
    def Invariantes(self):
        C = self.Cauchy_green_left()
        Invariantes = np.ones((len(C),3))
        for it, c in enumerate(C):
            I1 = np.trace(c)
            I2 = 0.5 * ( np.trace(c)**2 - np.trace(np.matmul(c,c)) )
            I3 = np.linalg.det(c)

            Invariantes[it,:] = np.array((I1,I2,I3))

        return Invariantes
    
    def Yeoh(self,c1,c2,c3):
        invariantes  = self.Invariantes()
        energy = np.ones((len(invariantes)))

        for it,i in enumerate(invariantes):
            I1 ,  I2 , I3 = i[0], i[1], i[2] 
            energia  =  c1 * (I1 - 3) + c2*(I1-3)**2 + c3*(I1-3)**3
            energy[it]  =  energia

        
        return energy
    
    def Mooney(self,c1,c2,c3):
        invariantes  = self.Invariantes()
        energy = np.ones((len(invariantes)))

        for it,i in enumerate(invariantes):
            I1 ,  I2 , I3 = i[0], i[1], i[2] 
            energia  =  c1 * (I1 - 3) + c2*(I2-3) + c3*(I1-3)*(I2-3)
            energy[it]  =  energia
    
        return energy
    
    def Demiray(self,c1,c2):
        invariantes  = self.Invariantes()
        energy = np.ones((len(invariantes)))

        for it,i in enumerate(invariantes):
            I1 ,  I2 , I3 = i[0], i[1], i[2] 
            aux = I3**(-1/3)
            energia  =  (c1/c2) * (np.e**(c2*0.5*(I1*aux-3)) - 1 )
            energy[it]  =  energia
    
        return energy