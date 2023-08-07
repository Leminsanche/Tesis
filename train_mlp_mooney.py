import numpy as np
import torch

import matplotlib.pyplot as plt
from launch.mlp_model import *
  

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu' 

    print("The device available is :", device)

    modelo_mat = 'Mooney'
    file_data = 'SVD/Biaxial_Mooney'
    dataset = BiaxialLoader(file_data)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=6)

    mlp = MLP(8,8)
    mlp.to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4, weight_decay=1e-4)

    train_model(mlp, trainloader, optimizer, device=device, epochs=5000)

    path = 'mlp_models_dis2grad/' + 'mlp_model_' + modelo_mat +'.pt'

    torch.save(mlp.state_dict(),path)

    test_loader = BiaxialLoader(file_data,train=False)

    evaluation_test(test_loader, mlp, modelo = modelo_mat , file  = 'mlp_models_dis2grad/', device=device)

