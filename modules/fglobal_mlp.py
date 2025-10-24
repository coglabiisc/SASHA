import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable


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
    

def print_model_summary(model):
    table = PrettyTable(["Layer", "Input Dim", "Output Dim", "Param Count"])
    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count

            # Extract input/output dims
            shape = list(param.shape)
            if 'weight' in name and len(shape) == 2:
                input_dim, output_dim = shape[1], shape[0]
            else:
                input_dim = '-'
                output_dim = shape

            table.add_row([name, input_dim, output_dim, param_count])

    print("\nModel Summary for FGlobal:\n")
    print(table)
    print(f"\nTotal trainable parameters: {total_params:,}")


# Example usage
if __name__ == "__main__":
    model = FGlobal()
    print_model_summary(model)