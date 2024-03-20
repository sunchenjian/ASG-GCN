import torch
from torch.nn import Linear as Lin
from torch_geometric.nn import GATConv, ChebConv
import torch.nn.functional as F
from torch import nn
from dataloader import get_adj
import math

class My_gnn(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edge_dropout, hgc, lg, device):
        super(My_gnn, self).__init__()
        K = 3
        self.mood_dim = 33

        hidden = [hgc for i in range(lg)]
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(33)
        self.norm3 = nn.LayerNorm(self.mood_dim)
        self.device = device
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.softmax = nn.Softmax(dim=1)
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg

        # 自注意力权重
        self.query = nn.Linear(self.mood_dim, self.mood_dim, bias=False)
        self.key = nn.Linear(self.mood_dim, self.mood_dim, bias=False)

        self.GRU = nn.GRU(input_dim, input_dim, batch_first=True)
        self.GAT1 = GATConv(input_dim, input_dim, dropout=0.2, add_self_loops=True, bias=True)
        self.GAT2 = GATConv(input_dim, input_dim, dropout=0.2, add_self_loops=True, bias=True)

        self.gconv = nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i == 0 else hidden[i-1]
            self.gconv.append(ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias))
        cls_input_dim = sum(hidden)

        self.cls = nn.Sequential(
            torch.nn.Linear(cls_input_dim, 128),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            torch.nn.Linear(128, num_classes)
        )
        self.model_init()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True


    def forward(self, features, mood_feature, edge_index):

        x = self.norm1(features)
        x, hn = self.GRU(x)

        nodes_list = []
        for i in range(x.shape[0]):
            graph = self.GAT1(x[i], edge_index)
            graph = self.GAT2(graph, edge_index)
            graph = torch.mean(graph, dim=0)
            nodes_list.append(graph)
        features = torch.stack([node for node in nodes_list], 0)
        features = features.detach().cpu().numpy()


        #情绪特征的注意力系数计算
        mood_feature = self.norm3(mood_feature)
        proj_query = self.query(mood_feature)
        proj_key = self.key(mood_feature)
        energy = torch.matmul(proj_query, proj_key.T)  # proj_query: NxD   proj_key.T: DxN
        energy = energy / math.sqrt(self.mood_dim)
        attention = self.softmax(energy)
        attention = (attention - attention.min())/(attention.max() - attention.min())
        attention = attention.detach().cpu().numpy()

        edge_index, edgenet_input, mood_weight = get_adj(features, attention)
        mood_weight = torch.tensor(mood_weight, dtype=torch.float32).to(self.device)

        edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(self.device)
        features = torch.tensor(features, dtype=torch.float32).to(self.device)

        edge_weight = torch.squeeze((self.cos(edgenet_input[:, 0:edgenet_input.shape[1]//2], edgenet_input[:, edgenet_input.shape[1]//2:])+1)*0.5)
        c = 0.6
        edge_weight = c*edge_weight + (1-c)*mood_weight

        features = F.dropout(features, self.dropout, self.training)
        h = self.relu(self.gconv[0](features, edge_index, edge_weight))
        h0 = h
        for i in range(1, self.lg):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_weight))
            jk = torch.cat((h0, h), axis=1)
            h0 = jk

        logit = self.softmax(self.cls(jk))


        return logit