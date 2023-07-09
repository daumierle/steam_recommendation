import os
import json
import torch
import pandas as pd

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData


class SteamDataset:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_test_data(self):
        with open(os.path.join(self.data_path, "active_user_games_test.json"), "r",
                  encoding="utf-8") as user_game_test_file:
            user_games = json.load(user_game_test_file)
        return user_games

    def get_all_games(self):
        with open(os.path.join(self.data_path, "all_game_data.json"), "r", encoding="utf-8") as game_info_file:
            all_games = json.load(game_info_file)
        all_games = list(all_games.keys())
        return all_games

    def get_label_test(self):
        with open(os.path.join(self.data_path, "active_user_games_test.json"), "r",
                  encoding="utf-8") as user_game_test_file:
            user_games = json.load(user_game_test_file)

        label_data = dict()
        for uid, games in user_games.items():
            label_data[uid] = list(set(games["owned_games"]).difference(set(games["prev_owned_games"])))

        return label_data


class SteamGraphData:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_game_features(self, all_games, train_game_ids):
        with open(os.path.join(self.data_path, "all_genres.json"), "r", encoding="utf-8") as all_genre_file:
            genres = json.load(all_genre_file)

        game_feat = list()
        for game_id in train_game_ids:
            game_feat.append([1 if genre in all_games[game_id]['genres'] else 0 for genre in genres])
        game_feat = torch.as_tensor(game_feat, dtype=torch.float)

        return game_feat

    def process_data(self):
        with open(os.path.join(self.data_path, "active_user_games_train.json"), "r", encoding="utf-8") as user_game_file:
            user_games = json.load(user_game_file)

        with open(os.path.join(self.data_path, "game_data_train.json"), "r", encoding="utf-8") as train_game_file:
            train_games = json.load(train_game_file)
        train_game_ids = list(train_games.keys())

        with open(os.path.join(self.data_path, "all_game_data_extended.json"), "r", encoding="utf-8") as all_game_file:
            all_games = json.load(all_game_file)

        game_features = self.get_game_features(all_games, train_game_ids)

        # Mapping of unique games and users
        unique_user_id = list(user_games.keys())
        unique_user_id = pd.DataFrame(data={
            'userId': unique_user_id,
            'mappedID': pd.RangeIndex(len(unique_user_id)),
        })

        unique_game_id = train_game_ids
        unique_game_id = pd.DataFrame(data={
            'gameId': unique_game_id,
            'mappedID': pd.RangeIndex(len(unique_game_id)),
        })

        # User-game dataframe
        user_ids, game_ids = list(), list()
        for user_id, games in user_games.items():
            owned_games = games['owned_games']
            owned_games = [str(game) for game in owned_games if str(game) in train_game_ids]
            user_ids.extend([user_id] * len(owned_games))
            game_ids.extend(owned_games)
        user_games_df = pd.DataFrame({"userId": user_ids, "gameId": game_ids})

        # Perform merge to obtain the edges from users and movies:
        owned_user_id = pd.merge(user_games_df['userId'], unique_user_id,
                                 left_on='userId', right_on='userId', how='left')
        owned_user_id = torch.from_numpy(owned_user_id['mappedID'].values)

        owned_game_id = pd.merge(user_games_df['gameId'], unique_game_id,
                                 left_on='gameId', right_on='gameId', how='left')
        owned_game_id = torch.from_numpy(owned_game_id['mappedID'].values)

        # With this, we are ready to construct our `edge_index` in COO format
        # following PyG semantics:
        edge_index_user_to_game = torch.stack([owned_user_id, owned_game_id], dim=0)
        print()
        print("Final edge indices pointing from users to movies:")
        print("=================================================")
        print(edge_index_user_to_game.shape)

        return unique_user_id, unique_game_id, game_features, edge_index_user_to_game

    @staticmethod
    def steam_graph_data(unique_user_id, unique_game_id, game_features, edge_index_user_to_game):
        data = HeteroData()
        # Save node indices:
        data["user"].node_id = torch.arange(len(unique_user_id))
        data["game"].node_id = torch.arange(len(unique_game_id))
        # Add the node features and edge indices:
        data["game"].x = game_features
        data["user", "owned", "game"].edge_index = edge_index_user_to_game
        # We also need to make sure to add the reverse edges from games to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        data = T.ToUndirected()(data)

        return data