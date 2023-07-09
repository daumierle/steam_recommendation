import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader

from utils.steam_data import SteamDataset, SteamGraphData
from models.baselines import RandomModel
from models.gnn_basics import GNNModel
from utils.metrics import Metrics


def run_baselines(data_path, models):
    """
    Baseline models
    :param data_path: path to where data is stored
    :param models: list of baseline models ["random"]
    :return:
    """
    steam_dataset = SteamDataset(data_path)
    test_data = steam_dataset.get_test_data()
    test_labels = steam_dataset.get_label_test()
    all_game_data = steam_dataset.get_all_games()

    for model in models:
        if model == "random":
            recsys = RandomModel()
        else:
            raise NotImplementedError("Model not found!")

        test_preds = list()
        for uid, games in tqdm(test_data.items()):
            test_preds.append(recsys.forward(games, all_game_data))

        # Evaluate
        print(f"+++ Evaluation: {model} model +++")
        metrics = Metrics(test_preds, test_labels)
        metrics.evaluate([5, 10, 20])


def run_gnn_models(data_path, model, mode):
    """
    Graph-based models
    :param data_path: path to where data is stored
    :param model: graph-based models, i.e. GraphSAGE, GAT, GCN
    :param mode: list of either ['train', 'val', 'test']
    :return:
    """
    steam_graph = SteamGraphData(data_path)
    unique_user_id, unique_game_id, game_features, edge_index_user_to_game = steam_graph.process_data()
    data = steam_graph.steam_graph_data(unique_user_id, unique_game_id, game_features, edge_index_user_to_game)

    # For this, we first split the set of edges into training (80%), validation (10%), and testing edges (10%).
    # Across the training edges, we use 70% of edges for message passing, and 30% of edges for supervision.
    # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=("user", "owned", "game"),
        rev_edge_types=("game", "rev_owned", "user"),
    )
    train_data, val_data, test_data = transform(data)

    # Define seed edges
    edge_label_index = train_data["user", "owned", "game"].edge_label_index
    edge_label = train_data["user", "owned", "game"].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        edge_label_index=(("user", "owned", "game"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
        shuffle=True,
    )

    recsys = GNNModel(data, model, hidden_channels=64)
    recsys = recsys.to(device)
    optimizer = torch.optim.Adam(recsys.parameters(), lr=0.001)

    for epoch in range(1, 11):
        total_loss = total_examples = 0
        for sampled_data in tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = recsys(sampled_data)
            ground_truth = sampled_data["user", "owned", "game"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        required=True,
        help="The path where the xlsx file is stored.",
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_baselines(args.data_path, ["random"])
