
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split

class BiaxialLoader(Dataset):
    def __init__(self, data_dir="datos", train=True, seed=123):

        X = (np.loadtxt(data_dir+"/Coeficientes_Desplazmaientos.txt")).T
        y = (np.loadtxt(data_dir+"/Coeficientes_Gradientes.txt")).T
        x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=seed)
        
        if train:
            self.x = torch.tensor(x_train)
            self.y = torch.tensor(y_train)
        else:
            self.x = torch.tensor(x_test)
            self.y = torch.tensor(y_test)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]