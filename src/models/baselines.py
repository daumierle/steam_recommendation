import os
import json
import random
from tqdm import tqdm

from utils.data_utils import get_label_test
from utils.metrics import Metrics

random.seed(13)


class RandomModel:
    def __init__(self, data_path, topk):
        """
        A random recommendation system
        :param data_path: path to where data is stored
        :param topk: can either be int or list of int
        """
        self.data_path = data_path
        self.k = topk

        print("+++ Random Recommendation System +++")
        self.test_labels = get_label_test(self.data_path)
        self.test_preds = self.random_recsys()

        scores = self.evaluate()
        if type(self.k) == int:
            print(f"MAP@{self.k}:", scores[0])
            print(f"Recall@{self.k}:", scores[1])
            print(f"NDCG@{self.k}:", scores[2])
        else:
            for s, score in enumerate(scores):
                print(f"MAP@{self.k[s]}:", scores[s][0])
                print(f"Recall@{self.k[s]}:", scores[s][1])
                print(f"NDCG@{self.k[s]}:", scores[s][2])
                print("==========")

    def random_recsys(self):
        with open(os.path.join(self.data_path, "active_user_games_test.json"), "r", encoding="utf-8") as user_game_test_file:
            user_games = json.load(user_game_test_file)

        with open(os.path.join(self.data_path, "all_game_data.json"), "r", encoding="utf-8") as game_info_file:
            all_games = json.load(game_info_file)
        all_games = list(all_games.keys())

        recsys_results = list()
        for uid, games in tqdm(user_games.items()):
            prev_owned_games = [str(game) for game in games["prev_owned_games"]]
            new_games = list(set(all_games) - set(prev_owned_games))
            random.shuffle(new_games)
            recsys_results.append(new_games[:20])

        return recsys_results

    def evaluate(self):
        if type(self.k) == int:
            metrics = Metrics(self.test_preds, self.test_labels, self.k)
            return (metrics.map_at_k(), metrics.recall_at_k(), metrics.ndcg_at_k())
        else:
            all_metrics = list()
            for k_val in self.k:
                metrics = Metrics(self.test_preds, self.test_labels, k_val)
                all_metrics.append((metrics.map_at_k(), metrics.recall_at_k(), metrics.ndcg_at_k()))
            return all_metrics


if __name__ == "__main__":
    steam_path = "F:\\Research\\datasets\\steam"
    RandomModel(steam_path, [5, 10, 20])
