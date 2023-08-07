import numpy as np
import torch

import matplotlib.pyplot as plt

from mlp_model import MLP

from dataloader import BiaxialLoader
  

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

def evaluation_test(test_loader, model, bins_histogram=100, device="cpu", save_fig=True, verbose=True):
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
    plt.savefig("relative_error_model.pdf") if save_fig else plt.show()
    plt.close()

    plt.title("Relative Error Histogram Distribution")
    plt.xlabel("Relative error in [%]")
    plt.hist(relative_error*100, bins=bins_histogram)
    plt.savefig("relative_ histogram_error_model.pdf") if save_fig else plt.show()
    plt.close()

    if verbose:
        print(f"The MSE test: {mse} \n The mean realive error: {mre}")

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu' 

    print("The device available is :", device)

    dataset = BiaxialLoader()
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=6)

    mlp = MLP(8,8)
    mlp.to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4, weight_decay=1e-4)

    train_model(mlp, trainloader, optimizer, device=device, epochs=5000)

    torch.save(mlp.state_dict(),"mlp_model_mooney.pt")

    test_loader = BiaxialLoader(train=False)

    evaluation_test(test_loader, mlp, device=device)

