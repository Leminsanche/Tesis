from launch.Funciones import *
from launch.gradientes import Gradientes_nodales_Vulcan
from launch.vulcan_handler import VulcanHandler


class Vulcan():

    def __init__(self,ubicacion_casos = 'Casos/'):
        self.ubicacion_casos = ubicacion_casos

######################################################################################################################
    def Biaxial_Demiray(self,a,b,penal = 1000,verbose = True, fuerzas_flag = False):
        print('##### Modelo Demiray ####')
        ubicacion = self.ubicacion_casos + 'Biaxial_Demiray/'
    
        Nombre_salida = 'Demiray_test'    
        parametros = {'Cons1': a,'Cons2': b,'Penal': a*penal } 
        dat, geo, fix, file_msh = files_vulcan(ubicacion)

        if verbose == True:
            print('Consante 1: ', a)
            print('Consante 2: ', b)
            print('Penalizador: ', a*penal)

        caso1 = VulcanHandler([dat,geo,fix],Nombre_salida)
        caso1.run(parametros)
        disp, stress = get_results(caso1.pathToPos())

        print('Numero de pasos simulados: ', len(disp))
        gradientes_deformacion = Gradientes_nodales_Vulcan(file_msh,disp)
        print('Simulacion Finalizada')

        if verbose == True:
            print('J global: ', gradientes_deformacion [1] )


        dat_out = Nombre_salida + '.dat' 
        fix_out = Nombre_salida + '.fix'
        geo_out = Nombre_salida + '.geo'

        Borrar_file([dat_out,fix_out,geo_out])

        if fuerzas_flag == True:
            Fuerzas = Resultados_vulcan(caso1.pathToPos()).Fuerzas()
            return disp, stress, gradientes_deformacion, file_msh, Fuerzas
            

        return disp, stress, gradientes_deformacion, file_msh
    
######################################################################################################################
    def Biaxial_Demiray_20(self,a,b,penal = 1000,verbose = True, fuerzas_flag = False):
        print('##### Modelo Demiray ####')
        ubicacion = self.ubicacion_casos + 'Biaxial_Demiray20/'
    
        Nombre_salida = 'Demiray_test'    
        parametros = {'Cons1': a,'Cons2': b,'Penal': a*penal } 
        dat, geo, fix, file_msh = files_vulcan(ubicacion)

        if verbose == True:
            print('Consante 1: ', a)
            print('Consante 2: ', b)
            print('Penalizador: ', a*penal)

        caso1 = VulcanHandler([dat,geo,fix],Nombre_salida)
        caso1.run(parametros)
        disp, stress = get_results(caso1.pathToPos())
        print('Numero de pasos simulados: ', len(disp))
        gradientes_deformacion = Gradientes_nodales_Vulcan(file_msh,disp)
        print('Simulacion Finalizada')

        if verbose == True:
            print('J global: ', gradientes_deformacion [1])



        dat_out = Nombre_salida + '.dat' 
        fix_out = Nombre_salida + '.fix'
        geo_out = Nombre_salida + '.geo'

        Borrar_file([dat_out,fix_out,geo_out])

        if fuerzas_flag == True:
            Fuerzas = Resultados_vulcan(caso1.pathToPos()).Fuerzas()
            return disp, stress, gradientes_deformacion, file_msh, Fuerzas

        return disp, stress, gradientes_deformacion, file_msh
    
######################################################################################################################
    def Biaxial_Yeoh_20(self,a,b,c,penal = 1000,verbose = True, fuerzas_flag = False):
        print('####### Modelo Yeoh #######')


        ubicacion = self.ubicacion_casos + 'Biaxial_Yeoh20/'
    
        Nombre_salida = 'Yeoh_test'    
        parametros = {'Cons1': a,'Cons2': b,'Cons3': c,'Penal': penal*a }
        dat, geo, fix, file_msh = files_vulcan(ubicacion)

        if verbose == True:
            print('Consante 1: ', a)
            print('Consante 2: ', b)
            print('Consante 3: ', c)
            print('Penalizador: ', a*penal)



        caso1 = VulcanHandler([dat,geo,fix],Nombre_salida)
        caso1.run(parametros)
        disp, stress = get_results(caso1.pathToPos())
        print('Numero de pasos simulados: ', len(disp))
        gradientes_deformacion = Gradientes_nodales_Vulcan(file_msh,disp)
        print('Simulacion Finalizada')

        if verbose == True:
            print('J global: ', gradientes_deformacion [1])


        dat_out = Nombre_salida + '.dat' 
        fix_out = Nombre_salida + '.fix'
        geo_out = Nombre_salida + '.geo'

        Borrar_file([dat_out,fix_out,geo_out])

        if fuerzas_flag == True:
            Fuerzas = Resultados_vulcan(caso1.pathToPos()).Fuerzas()
            return disp, stress, gradientes_deformacion, file_msh, Fuerzas

        return disp, stress, gradientes_deformacion, file_msh
    
######################################################################################################################
    def Biaxial_Yeoh(self,a,b,c,penal = 1000,verbose = True, fuerzas_flag = False):
        print('####### Modelo Yeoh #######')


        ubicacion = self.ubicacion_casos + 'Biaxial_Yeoh/'
    
        Nombre_salida = 'Yeoh_test'    
        parametros = {'Cons1': a,'Cons2': b,'Cons3': c,'Penal': penal*a }
        dat, geo, fix, file_msh = files_vulcan(ubicacion)

        if verbose == True:
            print('Consante 1: ', a)
            print('Consante 2: ', b)
            print('Consante 3: ', c)
            print('Penalizador: ', a*penal)



        caso1 = VulcanHandler([dat,geo,fix],Nombre_salida)
        caso1.run(parametros)
        disp, stress = get_results(caso1.pathToPos())
        print('Numero de pasos simulados: ', len(disp))
        gradientes_deformacion = Gradientes_nodales_Vulcan(file_msh,disp)
        print('Simulacion Finalizada')

        if verbose == True:
            print('J global: ', gradientes_deformacion [1])


        dat_out = Nombre_salida + '.dat' 
        fix_out = Nombre_salida + '.fix'
        geo_out = Nombre_salida + '.geo'

        Borrar_file([dat_out,fix_out,geo_out])

        if fuerzas_flag == True:
            Fuerzas = Resultados_vulcan(caso1.pathToPos()).Fuerzas()
            return disp, stress, gradientes_deformacion, file_msh, Fuerzas

        return disp, stress, gradientes_deformacion, file_msh
    
######################################################################################################################
    def Biaxial_Mooney(self,a,b,c,penal = 10,verbose = True, fuerzas_flag = False):
        print('####### Modelo Mooney Rivlin #######')


        ubicacion = self.ubicacion_casos + 'Biaxial_Mooney/'
    
        Nombre_salida = 'Mooney_test'    
        parametros = {'Cons1': a,'Cons2': b,'Cons3': c,'Penal': penal*a }
        dat, geo, fix, file_msh = files_vulcan(ubicacion)

        if verbose == True:
            print('Consante 1: ', a)
            print('Consante 2: ', b)
            print('Consante 3: ', c)
            print('Penalizador: ', a*penal)



        caso1 = VulcanHandler([dat,geo,fix],Nombre_salida)
        caso1.run(parametros)
        disp, stress = get_results(caso1.pathToPos())
        print('Numero de pasos simulados: ', len(disp))
        gradientes_deformacion = Gradientes_nodales_Vulcan(file_msh,disp)
        print('Simulacion Finalizada')

        if verbose == True:
            print('J global: ', gradientes_deformacion [1])


        dat_out = Nombre_salida + '.dat' 
        fix_out = Nombre_salida + '.fix'
        geo_out = Nombre_salida + '.geo'

        Borrar_file([dat_out,fix_out,geo_out])
    
        if fuerzas_flag == True:
            Fuerzas = Resultados_vulcan(caso1.pathToPos()).Fuerzas()
            return disp, stress, gradientes_deformacion, file_msh, Fuerzas

        return disp, stress, gradientes_deformacion, file_msh
    
######################################################################################################################
    def Cubo_demiray(self,a,b,penal = 10000,verbose = True):
        print('##### Modelo Demiray ####')
        ubicacion = self.ubicacion_casos + 'cubochico/'
        #print(ubicacion)
        Nombre_salida = 'cubo_test'    
        parametros = {'Cons1': a,'Cons2': b,'Penal': a*penal } 
        dat, geo, fix, file_msh = files_vulcan(ubicacion)

        if verbose == True:
            print('Consante 1: ', a)
            print('Consante 2: ', b)
            print('Penalizador: ', a*penal)

        caso1 = VulcanHandler([dat,geo,fix],Nombre_salida)
        caso1.run(parametros)
        disp, stress = get_results(caso1.pathToPos())
        print('Numero de pasos simulados: ', len(disp))
        gradientes_deformacion = Gradientes_nodales_Vulcan(file_msh,disp)
        print('Simulacion Finalizada')

        dat_out = Nombre_salida + '.dat' 
        fix_out = Nombre_salida + '.fix'
        geo_out = Nombre_salida + '.geo'

        Borrar_file([dat_out,fix_out,geo_out])


        D = np.zeros([3*len(disp[0]),len(disp)])
        ite = 0

        for jt, j in enumerate(disp):
            D[:,ite] = j.reshape(-1)
            ite = ite +1

        T = np.zeros([6*len(disp[0]),len(disp)])
        ite = 0

        for jt, j in enumerate(stress):
            T[:,ite] = j.reshape(-1)
            ite = ite +1

        

        return D, T, gradientes_deformacion, file_msh


