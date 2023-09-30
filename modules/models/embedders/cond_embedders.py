
import torch.nn as nn
import torch 
from monai.networks.layers.utils import get_act_layer

class LabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_classes, emb_dim)

        # self.embedding = nn.Embedding(num_classes, emb_dim//4)
        # self.emb_net = nn.Sequential(
        #     nn.Linear(1, emb_dim),
        #     get_act_layer(act_name),
        #     nn.Linear(emb_dim, emb_dim)
        # )

    def forward(self, condition):
        c = self.embedding(condition) #[B,] -> [B, C]
        # c = self.emb_net(c)
        # c = self.emb_net(condition[:,None].float())
        # c = (2*condition-1)[:, None].expand(-1, self.emb_dim).type(torch.float32)
        return c


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


