import json

import sacred


default_config_file = "default.json"


def load_config(filename=default_config_file) -> dict:
    with open(filename, "r") as file:
        return json.load(file)


config = load_config()
