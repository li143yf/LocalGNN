import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import dgl
import dgl.nn as dglnn


class SAGE(nn.Module): 
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):         
            h = layer(sg, h)
            if l == 1:  
                h_2=h                  
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h,h_2  

    def inference(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        return h


