import torch
from torch import nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, in_features, out_features, bias=True, temperature=10, logit_scale=1.0,
                 hidden_dim=1024, n_hidden=3, dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        self.dropout_prob = dropout
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout_prob > 0:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        return F.softmax(x / self.temperature, dim=-1) * self.logit_scale