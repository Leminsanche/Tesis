import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        
        # TODO
        # It is possible to improve by giving a dictionary with the architecture of the MLP.

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim,128).double(),
            nn.ReLU().double(),
            nn.Linear(128,128).double(),
            nn.ReLU().double(),
            nn.Linear(128,128).double(),
            nn.ReLU().double(),
            nn.Linear(128,128).double(),
            nn.ReLU().double(),
            nn.Linear(128,output_dim).double()
        )

    def forward(self, x):
        return self.mlp(x).double()
    


class MetricLogger:
    # TODO It's necessary to implement a Logger Class or use a library as wadb.
    def __init__(self):
        self.loss = []
        

def train_model(model, trainloader, optimizer, epochs=5000, verbose=True, device="cpu", loss_function = torch.nn.MSELoss(),
                 print_every=40):

    # TODO
    # Implement the early stopping criteria saving the best model. 
    # IMPLEMENT THE WEIGTHED BASED ON THE DISTRIBUTION OF \SIGMA IN THE LOSS FUNCTION   
    for epoch in range(epochs): 

        if verbose:
            print(f'Starting epoch {epoch+1}') 

        current_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).to(device)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if verbose:
                if i % print_every == 0:
                    print('Loss after mini-batch %5d: %.5f' %
                        (i + 1, current_loss / print_every))
                    current_loss = 0.0

def evaluation_test(test_loader, model,modelo ='Sin_modelo',file  = 'mlp_models_dis2grad/', bins_histogram=100, device="cpu", save_fig=True, verbose=True):
    X, y_test = test_loader[:]                                  # test set
    X = X.to(device)
    y_pred = model(X).to("cpu")                                           # Prediction of the model
    
    dy = y_test.detach().numpy()-y_pred.detach().numpy()        # Absolute difference in numpy
    
    relative_error = np.linalg.norm(dy, axis=1)/np.linalg.norm(y_test.detach().numpy(), axis=1) 
    
    mse = (dy**2).mean()           # MSE
    mre = relative_error.mean()     # Mean realative error  

    plt.title("Relative error")
    plt.plot(relative_error*100,'ko', label="Relative Error [%]")
    plt.legend()
    nombre2 = file + 'relative_error_model'+modelo+'.pdf'
    plt.savefig(nombre2) if save_fig else plt.show()
    plt.close()

    plt.title("Relative Error Histogram Distribution")
    plt.xlabel("Relative error in [%]")
    plt.hist(relative_error*100, bins=bins_histogram)
    nombre = file + "relative_ histogram_error_model_"+modelo+'.pdf'
    plt.savefig(nombre) if save_fig else plt.show()
    plt.close()

    if verbose:
        print(f"The MSE test: {mse} \n The mean realive error: {mre}")
    



class BiaxialLoader(Dataset):
    def __init__(self, data_dir="datos",X_dir ="/Coeficientes_Desplazmaientos.txt" , Y_dir =  "/Coeficientes_Gradientes.txt",  train=True, seed=123):
        
        X = (np.loadtxt(data_dir + X_dir )).T
        y = (np.loadtxt(data_dir + Y_dir)).T
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