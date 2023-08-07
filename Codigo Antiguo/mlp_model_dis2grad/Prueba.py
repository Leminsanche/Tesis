import torch
import numpy as np
from mlp_model import MLP
from d2g_mlp_train_model import evaluation_test
from dataloader import BiaxialLoader


mlp = MLP(8,8)

mlp.load_state_dict(torch.load("mlp_model.pt"))