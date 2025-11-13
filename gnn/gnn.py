from torch_geometric.nn import GCNConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F 
    
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x
    
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels//heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, out_channels//heads, heads=heads, dropout=dropout)
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        return x

class LinkPred(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super(LinkPred, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.mlp(x)
    
class SmallMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super(SmallMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.mlp(x)

def get_link_feat(link_aggr, x1, x2):
    if link_aggr == 'cat':
        return torch.cat([x1, x2], dim=1)
    if link_aggr == 'hdot':
        return x1 * x2
    if link_aggr == 'sum':
        return x1 + x2