import os
import json


def check_active_users(data_path):
    with open(os.path.join(data_path, "new_user_games.json"), "r", encoding="utf-8") as new_user_game_file:
        user_games = json.load(new_user_game_file)

    active_users = list()
    for user_uid, games in user_games.items():
        if set(games["owned_games"]).difference(set(games["prev_owned_games"])):
            active_users.append(user_uid)
    print(f"Active users: {len(active_users)} / {len(list(user_games.keys()))}")


def combine_user_game_files(data_path):
    active_user_games = dict()
    for fn in os.listdir(data_path):
        if fn.startswith("new_user_games"):
            with open(os.path.join(data_path, fn), "r", encoding="utf-8") as new_user_game_file:
                user_games = json.load(new_user_game_file)

            for user_uid, games in user_games.items():
                if set(games["owned_games"]).difference(set(games["prev_owned_games"])):
                    active_user_games[user_uid] = games

    print(f"Active users: {len(list(active_user_games.keys()))}")
    with open(os.path.join(data_path, "active_user_games_test.json"), "w", encoding="utf-8") as active_user_game_file:
        json.dump(active_user_games, active_user_game_file)


if __name__ == "__main__":
    steam_path = "F:\\Research\\datasets\\steam\\steam_data"
    # check_active_users(steam_path)
    combine_user_game_files(steam_path)
