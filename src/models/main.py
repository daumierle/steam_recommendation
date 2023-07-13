import os
import argparse

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader

from utils.steam_data import SteamDataset, SteamGraphData
from models.baselines import RandomModel, PopularityModel
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
        test_preds = list()
        if model == "random":
            recsys = RandomModel()
            for uid, games in tqdm(test_data.items()):
                test_preds.append(recsys.forward(games, list(all_game_data.keys())))
        elif model == "popularity":
            recsys = PopularityModel(test_data, all_game_data, type_="recommendations")
            for uid, games in tqdm(test_data.items()):
                test_preds.append(recsys.forward(games))
        else:
            raise NotImplementedError("Model not found!")

        # Evaluate
        print(f"+++ Evaluation: {model} model +++")
        metrics = Metrics(test_preds, test_labels)
        metrics.evaluate([5, 10, 20])


def run_gnn_models(data_path, model, mode):
    """
    Graph-based models
    :param data_path: path to where data is stored
    :param model: graph-based models, i.e. GraphSAGE, GAT, GCN
    :param mode: either 'train' or 'test'
    :return:
    """
    steam_graph = SteamGraphData(data_path)
    if not os.path.exists(os.path.join(data_path, "train_data.pt")):
        unique_user_id, unique_game_id, game_features, train_edge_index_user_to_game, test_edge_index_user_to_game = steam_graph.process_data()

        train_data = steam_graph.steam_graph_data(unique_user_id, unique_game_id, game_features,
                                                  train_edge_index_user_to_game)
        test_data = steam_graph.steam_graph_data(unique_user_id, unique_game_id, game_features,
                                                 test_edge_index_user_to_game)
        torch.save(train_data, os.path.join(data_path, "train_data.pt"))
        torch.save(test_data, os.path.join(data_path, "test_data.pt"))
    else:
        train_data = torch.load(os.path.join(data_path, "train_data.pt"), map_location=torch.device('cpu'))
        test_data = torch.load(os.path.join(data_path, "test_data.pt"), map_location=torch.device('cpu'))

    # Saved data path
    saved_model_path = os.path.join(data_path, f"graph_model/{model}")
    if not os.path.isdir(saved_model_path):
        os.makedirs(saved_model_path)

    # Transform: - for train_data: training (80%), validation (20%), test (0%)
    #            - for test_data: training (100%), validation (0%), test (0%)
    # Note: training in test_data is actually test samples
    if mode == "train":
        num_val = 0.2
        disjoint_train_ratio = 0.3
    else:
        num_val = 0.0
        disjoint_train_ratio = 0.0

    transform = T.RandomLinkSplit(
        num_val=num_val,
        num_test=0.0,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=("user", "owned", "game"),
        rev_edge_types=("game", "rev_owned", "user")
    )

    if mode == "train":
        train_data, val_data, _ = transform(train_data)

        # Train DataLoader
        edge_label_index = train_data["user", "owned", "game"].edge_label_index
        edge_label = train_data["user", "owned", "game"].edge_label
        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=[20, 10],
            neg_sampling_ratio=2.0,
            edge_label_index=(("user", "owned", "game"), edge_label_index),
            edge_label=edge_label,
            batch_size=128,
            shuffle=True
        )

        # Val DataLoader
        edge_label_index = val_data["user", "owned", "game"].edge_label_index
        edge_label = val_data["user", "owned", "game"].edge_label

        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors=[20, 10],
            edge_label_index=(("user", "owned", "game"), edge_label_index),
            edge_label=edge_label,
            batch_size=3 * 128,
            shuffle=False
        )

        recsys = GNNModel(train_data['user'].num_nodes, train_data['game'].num_nodes, train_data.metadata(),
                          model, hidden_channels=64)
        recsys = recsys.to(device)
        optimizer = torch.optim.Adam(recsys.parameters(), lr=0.001)
        best_val_auc = 0

        for epoch in range(1, 11):
            total_loss = total_examples = 0
            for sampled_data in tqdm(train_loader, desc="Train"):
                optimizer.zero_grad()
                sampled_data.to(device)
                pred = recsys(sampled_data)
                ground_truth = sampled_data["user", "owned", "game"].edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            print(f"+++ Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

            # Eval on validation set (using area under the curve score)
            val_preds, val_ground_truths = list(), list()
            for val_sample in tqdm(val_loader, desc="Eval"):
                with torch.no_grad():
                    val_sample.to(device)
                    val_preds.append(recsys(val_sample))
                    val_ground_truths.append(val_sample["user", "owned", "game"].edge_label)

            val_preds = torch.cat(val_preds, dim=0).cpu().numpy()
            val_ground_truths = torch.cat(val_ground_truths, dim=0).cpu().numpy()
            auc = roc_auc_score(val_ground_truths, val_preds)
            print(f"Validation AUC: {auc:.4f}")

            # Save best model
            if auc > best_val_auc:
                print(f"=== Save new best model @ epoch {epoch} ===")
                torch.save(recsys.state_dict(), os.path.join(saved_model_path, "recsys_model.ckpt"))
                best_val_auc = auc

    else:
        test_labels = steam_graph.get_label_test()

        test_data, _, _ = transform(test_data)

        # Load trained recsys model
        recsys = GNNModel(test_data['user'].num_nodes, test_data['game'].num_nodes, test_data.metadata(),
                          model, hidden_channels=64)
        recsys = recsys.to(device)
        recsys.load_state_dict(torch.load(os.path.join(saved_model_path, "recsys_model.ckpt")))
        recsys.eval()

        # Test DataLoader
        edge_label_index = test_data["user", "owned", "game"].edge_label_index
        edge_label = test_data["user", "owned", "game"].edge_label

        test_loader = LinkNeighborLoader(
            data=test_data,
            num_neighbors=[20, 10],
            edge_label_index=(("user", "owned", "game"), edge_label_index),
            edge_label=edge_label,
            batch_size=3 * 128,
            shuffle=False
        )

        test_preds = list()
        for test_sample in tqdm(test_loader, desc="Test"):
            with torch.no_grad():
                test_sample.to(device)
                test_preds.append(recsys(test_sample))
        print(len(test_labels), len(test_preds), test_preds[0])

        # Evaluate
        print(f"+++ Evaluation: {model} model +++")
        metrics = Metrics(test_preds, test_labels)
        metrics.evaluate([5, 10, 20])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        required=True,
        help="The path where the xlsx file is stored.",
    )
    parser.add_argument("--method", default="baseline", type=str, help="Model type: baseline, graph")
    parser.add_argument("--mode", default="train", type=str, help="train or test; not applicable to baseline")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.method == "baseline":
        run_baselines(args.data_path, ["random", "popularity"])
    elif args.method == "graph":
        run_gnn_models(args.data_path, "GraphSAGE", args.mode)
    else:
        raise NotImplementedError("Model type not found!")
