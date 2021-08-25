import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Color:
    DODGER_BLUE = "#0080FF"
    RED = '#FF0000'
    LIGHT_GREY = '#D0D0D0'
    NERO = '#202020'  # light black


def read_file(filename):
    content = []
    with open(filename, 'r') as file:
        for row in file:
            content.append(row.strip().lower())
    return content


class Board:
    @staticmethod
    def _generate_assassin_card(game_words):
        return {
            "id": 25,
            "name": game_words[-1],
            "type": "assassin",
            "colour": Color.NERO,
            "active": False,
        }

    @staticmethod
    def _generate_neutral_cards(game_words):
        return [
            {
                "id": game_words.index(word) + 1,
                "name": word,
                "type": "neutral",
                "colour": Color.LIGHT_GREY,
                "active": False
            }
            for word in game_words[17:24]
        ]

    @staticmethod
    def _generate_red_cards(game_words):
        return [
            {
                "id": game_words.index(word) + 1,
                "name": word,
                "type": "red",
                "colour": Color.RED,
                "active": False
            }
            for word in game_words[9:17]
        ]

    @staticmethod
    def _generate_blue_cards(game_words):
        return [
            {
                "id": game_words.index(word) + 1,
                "name": word,
                "type": "blue",
                "colour": Color.DODGER_BLUE,
                "active": False
            }
            for word in game_words[:9]
        ]

    def __init__(self, words_file):
        self.words_file = words_file
        self._board = self._generate_board()

    @property
    def board(self):
        return self._board

    def _generate_board(self):
        all_words = read_file(self.words_file)
        game_words = random.sample(all_words, 25)
        board = self._build_board(game_words)
        random.shuffle(board)  # rearrange game_words in frontend
        return board

    def _build_board(self, game_words):
        blues = self._generate_blue_cards(game_words)
        reds = self._generate_red_cards(game_words)
        neutrals = self._generate_neutral_cards(game_words)
        assassin = [self._generate_assassin_card(game_words)]  # to make possible concatenate all list cards
        return blues + reds + neutrals + assassin