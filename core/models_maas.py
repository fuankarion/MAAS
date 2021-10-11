import torch
import torch.nn as nn
import torch.nn.parameter

from torch_geometric.nn import EdgeConv, DynamicEdgeConv


class LinearPathPreact(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LinearPathPreact, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu(x1)
        x1 = self.fc1(x1)

        return x1


class MAAS(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, k):
        super(MAAS, self).__init__()
        self.dim_reduction_a = nn.Linear(in_feats, hidden_size)
        self.dim_reduction_v = nn.Linear(in_feats, hidden_size)

        self.next1 = EdgeConv(LinearPathPreact(hidden_size*2, hidden_size))
        self.next2 = EdgeConv(LinearPathPreact(hidden_size*2*2, hidden_size))
        self.next3 = EdgeConv(LinearPathPreact(hidden_size*2*2, hidden_size))
        self.next4 = EdgeConv(LinearPathPreact(hidden_size*2*2, hidden_size))

        self.dyn_next1 = DynamicEdgeConv(LinearPathPreact(hidden_size*2, hidden_size), k)
        self.dyn_next2 = DynamicEdgeConv(LinearPathPreact(hidden_size*2, hidden_size), k)
        self.dyn_next3 = DynamicEdgeConv(LinearPathPreact(hidden_size*2, hidden_size), k)
        self.dyn_next4 = DynamicEdgeConv(LinearPathPreact(hidden_size*2, hidden_size), k)

        self.fc = nn.Linear(hidden_size*2, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Audio preproc
        x0_a = self.dim_reduction_a(x)
        x0 = self.dim_reduction_v(x)
        x0[0::5, :] = x0_a[0::5, :].clone()

        # Dual Stream
        x1_dyn = self.dyn_next1(x0, batch)
        x2_dyn = self.dyn_next2(x1_dyn, batch)
        x3_dyn = self.dyn_next3(x2_dyn, batch)
        x4_dyn = self.dyn_next4(x3_dyn, batch)

        # Static Stream
        x1 = self.next1(x0, edge_index)
        x1_cat = torch.cat([x1_dyn, x1], dim=1)

        x2 = self.next2(x1_cat, edge_index)
        x2_cat = torch.cat([x2_dyn, x2], dim=1)

        x3 = self.next3(x2_cat, edge_index)
        x3_cat = torch.cat([x3_dyn, x3], dim=1)

        x4 = self.next4(x3_cat, edge_index)
        x4_cat = torch.cat([x4_dyn, x4], dim=1)

        return self.fc(x4_cat)
