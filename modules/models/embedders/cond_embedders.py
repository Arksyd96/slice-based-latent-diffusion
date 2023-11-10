import torch.nn as nn

class ConditionMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=512):
        super(ConditionMLP, self).__init__()
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_features),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, conditions):
        conditions = conditions.view(conditions.shape[0], -1)  # Flatten
        embedded_conditions = self.mlp(conditions)
        return embedded_conditions


