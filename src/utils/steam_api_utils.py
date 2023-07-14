import argparse
import json
import os
import re
import requests
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


def get_new_game_info(data_path, cont=0, mode="train"):
    with open(os.path.join(data_path, f"active_user_games_{mode}.json"), "r", encoding="utf-8") as user_game:
        user_data = json.load(user_game)

    if mode == "train":
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

    if mode == "test":
        game_test = []
        for user_id in user_data.keys():
            game = user_data[user_id]["owned_games"]
            game_test.extend(game)
        search_game = list(set(game_test).difference(search_game))

    game_data = {}
    unlisted_game = []

    for g in tqdm(range(cont, len(search_game))):
        try:
            info_dict = steam.apps.get_app_details(search_game[g])

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

                game = {str(search_game[g]): {"name": name, "price": price, "developers": developers,
                                              "publishers": publishers, "release_date": release_date,
                                              "genres": genres, "metacritic_score": metacritic}}
                game_data.update(game)
        except:
            with open(os.path.join(data_path, f"new_game_info_{cont}.json"), "w", encoding="utf-8") as new_game_info:
                json.dump(game_data, new_game_info)
            raise Exception("Error Occurred:", g)

    with open(os.path.join(data_path, f"new_game_info_{mode}_{cont}.json"), "w", encoding="utf-8") as new_game_info:
        json.dump(game_data, new_game_info)

    with open(os.path.join(data_path, f"unlisted_game_{mode}.json"), "w", encoding="utf-8") as unlisted_game_id:
        json.dump(unlisted_game, unlisted_game_id)


def add_data_field(data_path, cont=0):
    with open(os.path.join(data_path, "all_game_data_extended.json"), "r", encoding="utf-8") as existing_game_data:
        game_data = json.load(existing_game_data)

    for game_id in tqdm(list(game_data.keys())[cont:]):
        if game_data[game_id]["recommendations"] is None and game_data[game_id]["detailed_description"] == "" and \
                game_data[game_id]["about_the_game"] == "" and game_data[game_id]["header_image"] == "":
            try:
                info_dict = steam.apps.get_app_details(game_id)

                if info_dict and "data" in info_dict[game_id] and "recommendations" in info_dict[game_id]["data"]:
                    game_data[game_id]["recommendations"] = info_dict[game_id]["data"]["recommendations"]["total"]
                else:
                    game_data[game_id]["recommendations"] = None

                if info_dict and "data" in info_dict[game_id] and "detailed_description" in info_dict[game_id]["data"]:
                    game_data[game_id]["detailed_description"] = info_dict[game_id]["data"]["detailed_description"]
                else:
                    game_data[game_id]["detailed_description"] = ""

                if info_dict and "data" in info_dict[game_id] and "about_the_game" in info_dict[game_id]["data"]:
                    game_data[game_id]["about_the_game"] = info_dict[game_id]["data"]["about_the_game"]
                else:
                    game_data[game_id]["about_the_game"] = ""

                if info_dict and "data" in info_dict[game_id] and "header_image" in info_dict[game_id]["data"]:
                    game_data[game_id]["header_image"] = info_dict[game_id]["data"]["header_image"]
                else:
                    game_data[game_id]["header_image"] = ""

            except:
                with open(os.path.join(data_path, f"all_game_data_extended_{cont}.json"), "w",
                          encoding="utf-8") as new_game_data:
                    json.dump(game_data, new_game_data)
                raise Exception("Error Occurred:", game_id)

    with open(os.path.join(data_path, f"all_game_data_extended_{cont}.json"), "w", encoding="utf-8") as new_game_data:
        json.dump(game_data, new_game_data)


def get_all_games(data_path):
    with open(os.path.join(data_path, "all_app_ids.json"), "r", encoding="utf-8") as all_app_file:
        data = json.load(all_app_file)

    all_game_data = {"coming_soon": dict(), "no_date": dict(), "before_2000": dict()}
    unlisted_game = []
    error_games = []
    request_exception = 0

    for app in tqdm(data):
        try:
            info_dict = steam.apps.get_app_details(app['appid'])

            appid = str(app['appid'])
            if info_dict is None or not info_dict or "data" not in info_dict[appid] or not \
                    info_dict[appid]['data']:
                unlisted_game.append(app['appid'])
            else:
                if info_dict[appid]['data']['type'] == "game":
                    name = info_dict[appid]["data"]["name"]
                    if info_dict[appid]["data"]["is_free"]:
                        price = 0
                    else:
                        price = round(info_dict[appid]["data"]["price_overview"]["final"] / 100 / 23000,
                                      2) if "price_overview" in info_dict[appid]["data"] else None

                    developers = info_dict[appid]["data"]["developers"] if "developers" in info_dict[appid][
                        "data"] else ""
                    publishers = info_dict[appid]["data"]["publishers"] if "publishers" in info_dict[appid][
                        "data"] else ""

                    if "date" in info_dict[appid]["data"]["release_date"]:
                        release_date = info_dict[appid]["data"]["release_date"]["date"].strip()
                    elif "coming_soon" in info_dict[appid]["data"]["release_date"]:
                        release_date = "coming_soon"
                    else:
                        release_date = ""

                    year = re.findall("20\d{2}$", release_date)
                    if year and year[0] not in all_game_data:
                        all_game_data[year[0]] = dict()

                    genres = [desc["description"] for desc in info_dict[appid]["data"]["genres"]] if "genres" in \
                                                                                                     info_dict[appid][
                                                                                                         "data"] else []
                    metacritic = info_dict[appid]["data"]["metacritic"]["score"] if "metacritic" in info_dict[appid][
                        "data"] else None
                    recommendations = info_dict[appid]["data"]["recommendations"]["total"] if "recommendations" in \
                                                                                              info_dict[appid][
                                                                                                  "data"] else None

                    detailed_description = info_dict[appid]["data"]["detailed_description"] if "detailed_description" in \
                                                                                               info_dict[appid][
                                                                                                   "data"] else ""
                    about_the_game = info_dict[appid]["data"]["about_the_game"] if "about_the_game" in info_dict[appid][
                        "data"] else ""
                    header_image = info_dict[appid]["data"]["header_image"] if "header_image" in info_dict[appid][
                        "data"] else ""

                    if not release_date:
                        all_game_data["no_date"][appid] = {"name": name, "price": price, "developers": developers,
                                                           "publishers": publishers, "release_date": release_date,
                                                           "genres": genres, "metacritic_score": metacritic,
                                                           "recommendations": recommendations,
                                                           "detailed_description": detailed_description,
                                                           "about_the_game": about_the_game,
                                                           "header_image": header_image}
                    elif release_date == "coming_soon":
                        all_game_data[release_date][appid] = {"name": name, "price": price, "developers": developers,
                                                              "publishers": publishers, "release_date": release_date,
                                                              "genres": genres, "metacritic_score": metacritic,
                                                              "recommendations": recommendations,
                                                              "detailed_description": detailed_description,
                                                              "about_the_game": about_the_game,
                                                              "header_image": header_image}
                    else:
                        if year:
                            all_game_data[year[0]][appid] = {"name": name, "price": price,
                                                             "developers": developers,
                                                             "publishers": publishers,
                                                             "release_date": release_date,
                                                             "genres": genres, "metacritic_score": metacritic,
                                                             "recommendations": recommendations,
                                                             "detailed_description": detailed_description,
                                                             "about_the_game": about_the_game,
                                                             "header_image": header_image}
                        else:
                            all_game_data["before_2000"][appid] = {"name": name, "price": price,
                                                                   "developers": developers,
                                                                   "publishers": publishers,
                                                                   "release_date": release_date,
                                                                   "genres": genres, "metacritic_score": metacritic,
                                                                   "recommendations": recommendations,
                                                                   "detailed_description": detailed_description,
                                                                   "about_the_game": about_the_game,
                                                                   "header_image": header_image}

        except requests.exceptions.RequestException as e:
            request_exception += 1
            error_games.append(app['appid'])
            if request_exception >= 10:
                break
            else:
                continue

        except:
            error_games.append(app['appid'])
            continue

    for rel_year, val in all_game_data.items():
        with open(os.path.join(data_path, f"{rel_year}_games.json"), "w", encoding="utf-8") as year_file:
            json.dump(val, year_file)

    with open(os.path.join(data_path, "error_games.json"), "w", encoding="utf-8") as error_file:
        json.dump(error_games, error_file)


if __name__ == "__main__":
    steam = Steam(KEY)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steam_path",
        default=None,
        type=str,
        required=True,
        help="The path where the xlsx file is stored.",
    )
    args = parser.parse_args()

    # get_active_users(args.steam_path, cont=10023)
    # get_new_game_info(args.steam_path, cont=0, mode="test")
    # add_data_field(args.steam_path, cont=6044)
    get_all_games(args.steam_path)
