import torch
import numpy as np

from mlp_model import MLP

from d2g_mlp_train_model import evaluation_test

from dataloader import BiaxialLoader



if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu' 

    # Model use
    mlp = MLP(8,8).to(device)

    file = 'mlp_model_dis2grad/'+'mlp_model.pt'
    mlp.load_state_dict(torch.load(file)) # Loading the model weights.

    # To use the mlp function, you need to give a torch tensor with shape (1,8)
    # if you want to calculate multiple exampleas in one batch, enter the values in a vertical stack of shape (batch_number, 8)
    # the output is a tensor with shape (number_batch, 8)
    # In order to convert to numpy, you need to apply something like this y = mlp(x), y = y.detach().cpu().numpy()

    # The default option for torch is to handle float32, the models is trained as double or float64, so it is necessary to state that.

    y = mlp(torch.rand(2,8).double().to(device)).to(device)

    print(f"type:{type(y)}, shape:{y.shape}, dtype:{y.dtype}")

    y = y.detach().cpu().numpy()

    print(f"type:{type(y)}, shape:{y.shape}, dtype:{y.dtype}")

    # Here we are calling the MLP to show the test metrics of error.

    dataset = BiaxialLoader(train=False) # If not train==True then test

    test_loader = BiaxialLoader(train=False)

    evaluation_test(dataset, mlp, device=device, save_fig=True)
