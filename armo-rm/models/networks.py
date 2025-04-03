import torch
from torch import nn
import torch.nn.functional as F

class ScoreProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, pos):
        out = self.proj(pos)
        return out
    
class BetaHead(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.alpha_beta_proj = nn.Linear(output_dim, 2 * output_dim)
        self.softplus = nn.Softplus()

    def forward(self, pos_out, neg_out):
        diff = pos_out - neg_out
        alpha_beta = self.alpha_beta_proj(diff)
        alpha, beta = alpha_beta.chunk(2, dim=-1)
        alpha = self.softplus(alpha) + 1e-4
        beta = self.softplus(beta) + 1e-4
        return alpha, beta



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