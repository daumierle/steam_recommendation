import os
import json


def get_genre(data_path):
    with open(os.path.join(data_path, "Games_Genres.txt"), "r", encoding="utf-8") as genre_file:
        genre_info = genre_file.readlines()

    genre = dict()
    for line in genre_info:
        line = line.replace("\n", "")
        line_data = line.split(",")
        if line_data[0] not in genre:
            genre[line_data[0]] = [line_data[1]]
        else:
            genre[line_data[0]].append(line_data[1])

    return genre


def get_publisher(data_path):
    with open(os.path.join(data_path, "Games_Publishers.txt"), "r", encoding="utf-8") as publisher_file:
        publisher_info = publisher_file.readlines()

    publisher = dict()
    for line in publisher_info:
        line = line.replace("\n", "")
        line_data = line.split(",")
        publisher[line_data[0]] = line_data[1]

    return publisher


def get_developer(data_path):
    with open(os.path.join(data_path, "Games_Developers.txt"), "r", encoding="utf-8") as developer_file:
        developer_info = developer_file.readlines()

    developer = dict()
    for line in developer_info:
        line = line.replace("\n", "")
        line_data = line.split(",")
        developer[line_data[0]] = line_data[1]

    return developer


def get_game_info(data_path):
    with open(os.path.join(data_path, "App_ID_Info.txt"), "r", encoding="utf-8") as app_info_file:
        app_info = app_info_file.readlines()

    genre = get_genre(data_path)
    developer = get_developer(data_path)
    publisher = get_publisher(data_path)

    game_info = dict()
    for line in app_info:
        line = line.replace("\n", "")
        line_data = line.split(",")
        if line_data[2] == "game":
            game_info[line_data[0]] = {"name": line_data[1], "price": line_data[3],
                                       "developer": developer[line_data[0]] if line_data[0] in developer else "",
                                       "publisher": publisher[line_data[0]] if line_data[0] in publisher else "",
                                       "release_date": line_data[4].split()[0]}

    for id_, game in game_info.items():
        if id_ in genre:
            game_info[id_]["genres"] = genre[id_]
        else:
            game_info[id_]["genres"] = []

    with open(os.path.join(data_path, "game_info.json"), "w", encoding="utf-8") as game_info_file:
        json.dump(game_info, game_info_file)


def get_user_prev_games(data_path):
    with open(os.path.join(data_path, "new_user_games.json"), "r", encoding="utf-8") as new_user_game_file:
        user_games = json.load(new_user_game_file)
    user_uids = list(user_games.keys())

    with open(os.path.join(data_path, "train_game.txt"), "r", encoding="utf-8") as user_game_file:
        for line in user_game_file:
            if not user_uids:
                break
            line = line.replace("\n", "")
            line_data = line.split(",")
            if line_data[0] in user_games:
                user_games[line_data[0]]['prev_owned_games'] = [int(val) for val in line_data[1:]]
                user_uids.remove(line_data[0])

    with open(os.path.join(data_path, "new_user_games.json"), "w", encoding="utf-8") as user_game_file:
        json.dump(user_games, user_game_file)


def get_game_data_test(data_path):
    with open(os.path.join(data_path, "active_user_games_test.json"), "r", encoding="utf-8") as user_game_file:
        user_games = json.load(user_game_file)

    with open(os.path.join(data_path, "game_data_train.json"), "r", encoding="utf-8") as train_game_file:
        train_games = json.load(train_game_file)

    with open(os.path.join(data_path, "new_game_data_test.json"), "r", encoding="utf-8") as test_game_file:
        test_games = json.load(test_game_file)

    all_test_games = list()
    for uid, games in user_games.items():
        all_test_games.extend([str(game) for game in games['owned_games']])
    all_test_games = list(set(all_test_games))

    for game_id in all_test_games:
        if game_id in train_games and game_id not in test_games:
            test_games[game_id] = train_games[game_id]

    with open(os.path.join(data_path, "game_data_test.json"), "w", encoding="utf-8") as all_test_game_file:
        json.dump(test_games, all_test_game_file)


def get_all_games(data_path):
    with open(os.path.join(data_path, "game_data_train.json"), "r", encoding="utf-8") as game_train_file:
        train_games = json.load(game_train_file)

    with open(os.path.join(data_path, "new_game_data_test.json"), "r", encoding="utf-8") as game_test_file:
        test_games = json.load(game_test_file)

    all_games = train_games
    all_games.update(test_games)

    with open(os.path.join(data_path, "all_game_data.json"), "w", encoding="utf-8") as all_game_file:
        json.dump(all_games, all_game_file)


def get_all_genres(data_path):
    with open(os.path.join(data_path, "all_game_data_extended.json"), "r", encoding="utf-8") as all_game_file:
        all_games = json.load(all_game_file)

    genres = list()
    for game_id, game_details in all_games.items():
        genres.extend(game_details['genres'])
    genres = list(set(genres))

    with open(os.path.join(data_path, "all_genres.json"), "w", encoding="utf-8") as all_genre_file:
        json.dump(genres, all_genre_file)


if __name__ == "__main__":
    steam_path = "F:\\Research\\datasets\\steam"
    # get_game_info(steam_path)
    # get_user_prev_games(steam_path)
    # get_all_genres(steam_path)
    get_game_data_test(steam_path)
    # get_all_games(steam_path)
