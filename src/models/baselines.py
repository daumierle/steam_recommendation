import random
import pandas as pd
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
    def __init__(self, game_data, type_):
        self.type = type_
        self.game_data = game_data
        pass

    def game_list(self):
        game_suggestion = dict()
        for k, values in self.game_data.items():
            for v in values[self.type]:
                game_suggestion.setdefault(v, []).append(k)
        return game_suggestion

    def forward(self, user_games):
        prev_owned_games = [str(game) for game in user_games['prev_owned_games']]
        favorite_genre = Counter([genre for game_id, games in self.game_data.items() for genre in games[self.type]
                                  if game_id in prev_owned_games])
        top_genre = sorted(dict(favorite_genre), reverse=True)[0]
        new_games = [game for game in self.game_list()[top_genre] if game not in prev_owned_games]
        random.shuffle(new_games)
        return new_games[:20]


class ContentBasedModel:
    def __init__(self):
        pass


class ClusteringModel:
    def __init__(self, game_data):
        self.game_data = game_data
        pass

    def dict_to_df(self):
        data = pd.DataFrame(self.game_data)
        return data

    def forward(self):


