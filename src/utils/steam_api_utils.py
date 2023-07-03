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


def get_new_game_info(data_path, cont=0):

    with open(os.path.join(data_path, "active_user_games.json"), "r", encoding="utf-8") as user_game:
        user_data = json.load(user_game)

    if not os.path.exists(os.path.join(data_path, "unique_game_key.json")):
        game_id = []
        for i in range(len(list(user_data.keys()))):
            game_list = list(list(user_data.values())[i].values())[0]
            game_id.append(game_list)
        game_key = [b for a in game_id for b in a]
        unique_game_key = list(set(game_key))

        with open(os.path.join(data_path, "unique_game_key.json"), "w", encoding="utf-8") as unique_game_id:
            json.dump(unique_game_key, unique_game_id)

    with open(os.path.join(data_path, "unique_game_key.json"), "r", encoding="utf-8") as unique_game_id_list:
        search_game = json.load(unique_game_id_list)
    # print(steam.apps.get_app_details(search_game[1798]))

    game_data = {}
    unlisted_game = []

    for g in tqdm(range(cont, len(search_game))):
        try:
            game_info = steam.apps.get_app_details(search_game[g])
            info_dict = json.loads(game_info)

            if info_dict is None or "data" not in info_dict[str(search_game[g])]:
                unlisted_game.append(search_game[g])
            else:
                name = info_dict[str(search_game[g])]["data"]["name"]
                if info_dict[str(search_game[g])]["data"]["is_free"]:
                    price = 0
                elif "price_overview" not in info_dict[str(search_game[g])]["data"]:
                    price = None
                else:
                    price = round(info_dict[str(search_game[g])]["data"]["price_overview"]["final"] / 100 / 23000, 2)
                if "developers" not in info_dict[str(search_game[g])]["data"]:
                    developers = ""
                else:
                    developers = info_dict[str(search_game[g])]["data"]["developers"]
                if "publishers" not in info_dict[str(search_game[g])]["data"]:
                    publishers = ""
                else:
                    publishers = info_dict[str(search_game[g])]["data"]["publishers"]
                if info_dict[str(search_game[g])]["data"]["release_date"]["coming_soon"]:
                    release_date = ""
                else:
                    release_date = info_dict[str(search_game[g])]["data"]["release_date"]["date"]
                if "genres" not in info_dict[str(search_game[g])]["data"]:
                    genres = []
                else:
                    genres = [desc["description"] for desc in info_dict[str(search_game[g])]["data"]["genres"]]
                if "metacritic" in info_dict[str(search_game[g])]["data"]:
                    metacritic = info_dict[str(search_game[g])]["data"]["metacritic"]["score"]
                else:
                    metacritic = None

                game = {str(unique_game_key[g]): {"name": name, "price": price, "developers": developers,
                                                  "publishers": publishers, "release_date": release_date,
                                                  "genres": genres, "metacritic_score": metacritic}}
                game_data.update(game)
        except:
            with open(os.path.join(data_path, f"new_game_info_{cont}.json"), "w", encoding="utf-8") as new_game_info:
                json.dump(game_data, new_game_info)
            raise Exception("Error Occurred:", g)

    with open(os.path.join(data_path, f"new_game_info_{cont}.json"), "w", encoding="utf-8") as new_game_info:
        json.dump(game_data, new_game_info)

    with open(os.path.join(data_path, "unlisted_game.json"), "w", encoding="utf-8") as unlisted_game_id:
        json.dump(unlisted_game, unlisted_game_id)


def merge_dataset(data_path):

    all_game_data = {}

    for file in os.listdir(data_path):
        if file.startswith("new_game_info"):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as game_data:
                game_data_info = json.load(game_data)
            all_game_data.update(game_data_info)

    with open(os.path.join(data_path, "game_data_train.json"), "w", encoding="utf-8") as game_data_train:
        json.dump(all_game_data, game_data_train)


if __name__ == "__main__":
    steam = Steam(KEY)
    steam_path = "D:\\projects\\steam\\dataset"
    # get_active_users(steam_path, cont=10023)
    # get_new_game_info(steam_path, cont=9873)
    merge_dataset(steam_path)

