import torch
from torch import nn

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
    
