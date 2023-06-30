import json
import os
from tqdm import tqdm
from steam import Steam
from decouple import config

KEY = config("STEAM_API_KEY")


def get_active_users(data_path, cont=0):
    with open(os.path.join(data_path, "users.txt"), "r", encoding="utf-8") as users_file:
        users = users_file.readlines()

    user_games = dict()
    for line in tqdm(users[cont:]):
        line = line.replace("\n", "")
        try:
            recently_played_games = steam.users.get_user_recently_played_games(line)
            owned_games = steam.users.get_owned_games(line)
            if recently_played_games and owned_games:
                if 'games' in recently_played_games:
                    recently_played_games = [game['appid'] for game in recently_played_games['games']]
                else:
                    recently_played_games = []

                if 'games' in owned_games:
                    owned_games = [game['appid'] for game in owned_games['games']]
                else:
                    owned_games = []

                user_games[line] = {"owned_games": owned_games, "recently_played_games": recently_played_games}
        except:
            with open(os.path.join(data_path, f"new_user_games_{cont}.json"), "w", encoding="utf-8") as user_games_file:
                json.dump(user_games, user_games_file)
            raise Exception("Error occurred:", line)

    with open(os.path.join(data_path, f"new_user_games_{cont}.json"), "w", encoding="utf-8") as user_games_file:
        json.dump(user_games, user_games_file)


def get_new_game_info(data_path):
    with open(os.path.join(data_path, "game_info.json"), "r", encoding="utf-8") as game_info_file:
        game_data = json.load(game_info_file)


if __name__ == "__main__":
    steam = Steam(KEY)
    steam_path = "F:\\Research\\datasets\\steam\\steam_data"
    get_active_users(steam_path, cont=10023)
