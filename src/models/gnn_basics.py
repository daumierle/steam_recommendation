import torch
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv, HANConv, DeepGCNLayer, GENConv, to_hetero


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.gat_conv = GATv2Conv(hidden_channels, hidden_channels, heads=8, add_self_loops=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gat_conv(x, edge_index)
        return x


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = torch.nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = torch.nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=bool(i % 3))
            self.layers.append(layer)

    def forward(self, x, edge_index):
        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        return x


class HAN(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.han_conv = HANConv(hidden_channels, hidden_channels,
                                metadata=metadata, heads=8, dropout=0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.han_conv(x, edge_index)
        return x


class LinkClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x_user: torch.Tensor, x_movie: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Convert node embeddings to edge-level representations
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class GNNModel(torch.nn.Module):
    def __init__(self, num_user_nodes, num_game_nodes, metadata, model, hidden_channels, num_genres=74):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two embedding matrices for users and games
        self.game_lin = torch.nn.Linear(num_genres, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_user_nodes, hidden_channels)
        self.game_emb = torch.nn.Embedding(num_game_nodes, hidden_channels)
        # Instantiate homogeneous GNN
        if model == "GraphSAGE":
            self.gnn = GraphSAGE(hidden_channels)
        elif model == "GAT":
            self.gnn = GAT(hidden_channels)
        elif model == "GCN":
            self.gnn = GCN(hidden_channels)
        elif model == "DeeperGCN":
            self.gnn = DeeperGCN(hidden_channels, num_layers=16)
        elif model == "HAN":
            self.gnn = HAN(hidden_channels, metadata)
        else:
            raise NotImplementedError("Model not found!")
        # Convert GNN model into a heterogeneous variant
        self.gnn = to_hetero(self.gnn, metadata=metadata)
        self.classifier = LinkClassifier()

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {
            "user": self.user_emb(data["user"].node_id),
            "game": self.game_lin(data["game"].x) + self.game_emb(data["game"].node_id)
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["game"],
            data["user", "owned", "game"].edge_label_index
        )
        return pred
