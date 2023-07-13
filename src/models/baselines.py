import random
from collections import Counter

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


class PopularityModel:
    def __init__(self, all_user_games, game_data, type_):
        self.type = type_
        self.user_games = all_user_games
        self.game_data = game_data
        pass

    def sort_dict(self):
        if self.type == "recommendations":
            sorted_game_data = sorted(self.game_data.items(), key=lambda k: int(
                0 if k[1]["recommendations"] is None else k[1]["recommendations"]), reverse=True)
        elif self.type == "metacritic":
            sorted_game_data = sorted(self.game_data.items(), key=lambda k: int(
                0 if k[1]["metacritic_score"] is None else k[1]["metacritic_score"]), reverse=True)
        else:
            game_count = Counter(
                [str(game) for _, games in self.user_games.items() for game in games["prev_owned_games"]])
            sorted_game_data = sorted(dict(game_count).items(), key=lambda k: k[1], reverse=True)
        return dict(sorted_game_data)

    def forward(self, user_games):
        prev_owned_games = [str(game) for game in user_games['prev_owned_games']]
        new_games = dict()
        for game_id, games in self.sort_dict().items():
            if game_id not in prev_owned_games:
                new_games[game_id] = games

        if self.type == "recommendations":
            return list(dict(new_games).keys())[:20]
        elif self.type == "metacritic":
            return list(dict(new_games).keys())[:20]
        else:
            return list(dict(new_games).keys())[:20]


class SamenessModel:
    def __init__(self, type_):
        pass


class ContentBasedModel:
    def __init__(self):
        pass


class ClusteringModel:
    def __init__(self):
        pass

