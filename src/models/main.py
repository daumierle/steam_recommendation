import argparse
from tqdm import tqdm

from utils.data_utils import SteamDataset
from models.baselines import RandomModel
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

    run_baselines(args.data_path, ["random"])
