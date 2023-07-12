import torch
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
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
