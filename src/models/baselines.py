import random

random.seed(13)


class RandomModel:
    def __init__(self):
        pass

    @staticmethod
    def forward(user_games, all_games):
        prev_owned_games = [str(game) for game in user_games["prev_owned_games"]]
        new_games = list(set(all_games) - set(prev_owned_games))
        random.shuffle(new_games)
        return new_games[:20]

