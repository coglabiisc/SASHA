import torch
import torch.nn as nn
import torch.nn.functional as F


class FGlobal(nn.Module):
    def __init__(self, ip_dim=384*3, op_dim=384, hidden_dim=768):
        # call constructor from superclass
        super().__init__()
    
        # define network layers
        self.fc1 = nn.Linear(ip_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, op_dim)
        self.layer_norm = nn.LayerNorm(op_dim)

    def forward(self, x):
        # define forward pass
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x
    

    
def loaf_f_global(ckpt_path):
    mlp = FGlobal()#.cuda()
    mlp.load_state_dict(torch.load(ckpt_path))
    mlp.eval()
    return mlp