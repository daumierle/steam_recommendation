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
    def __init__(self, type_):
        self.type = type_
        pass

    def forward(self, user_games, all_user_games, game_data):
        prev_owned_games = [str(game) for game in user_games['prev_owned_games']]
        new_games = dict()
        for game_id, games in game_data.items():
            if game_id not in prev_owned_games:
                new_games[game_id] = games

        if self.type == "recommendations":
            sorted_new_games = sorted(new_games.items(), key=lambda k: int(0 if k[1]["recommendations"] is None else k[1]["recommendations"]), reverse=True)
            return list(dict(sorted_new_games).keys())[:20]
        elif self.type == "metacritic":
            sorted_new_games = sorted(new_games.items(), key=lambda k: int(0 if k[1]["metacritic_score"] is None else k[1]["metacritic_score"]), reverse=True)
            return list(dict(sorted_new_games).keys())[:20]
        else:
            game_count = Counter([str(game) for _, games in all_user_games.items()] for game in games["prev_owned_games"])
            new_game_count = dict()
            for game_id, games in dict(game_count).items():
                if game_id not in prev_owned_games:
                    new_game_count[game_id] = games
            sorted_new_games = sorted(new_game_count, reverse=True)
            return sorted_new_games[:20]


class SamenessModel:
    def __init__(self, type_):
        pass


class ContentBasedModel:
    def __init__(self):
        pass


class ClusteringModel:
    def __init__(self):
        pass

