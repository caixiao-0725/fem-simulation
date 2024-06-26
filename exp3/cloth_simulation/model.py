import torch
from torch_geometric.nn import GCNConv, GraphConv,global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import *

class GCN(torch.nn.Module):
    def __init__(self, n_features, hidden_channels,n_outputs):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(n_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, n_outputs*3)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        x = self.lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3):
        super(Decoder, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D-1)])
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.elu(h)
        outputs = self.output_linear(h)
        return outputs
    
class MDN3(nn.Module):
    def __init__(self):
        super(MDN3, self).__init__()
        self.encoder = GCN(n_features=6, hidden_channels=hidden_channels,n_outputs=feat_dim)
        self.decoder1 = Decoder(D=1, W=64, input_ch=feat_dim , output_ch=1)
        self.decoder2 = Decoder(D=1, W=64, input_ch=feat_dim , output_ch=1)
        self.decoder3 = Decoder(D=1, W=64, input_ch=feat_dim , output_ch=1)

    def forward(self, x,edge_index):
        interped = self.encoder(x,edge_index).view(x.shape[0],3,-1)
        decoded1 = self.decoder1(interped[..., [0], :])
        decoded2 = self.decoder2(interped[..., [1], :])
        decoded3 = self.decoder3(interped[..., [2], :])
        return torch.cat((decoded1, decoded2, decoded3), dim = 1)
        