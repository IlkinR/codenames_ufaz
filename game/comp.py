import random

import numpy as np


class Computer:
    """Generate a random sequence of computer moves"""

    def __init__(self, board):
        self.board = board
        self.distribution = self.get_distribution()
        self.blue, self.red, self.neutral = self.get_types()

    def get_types(self):
        """Extract the types from the cards"""

        not_actives = [card for card in self.board[1:] if not card["active"]]
        blues = [card['id'] for card in not_actives if card["type"] == "blue"]
        reds = [card['id'] for card in not_actives if card["type"] == "red"]
        neutrals = [card['id'] for card in not_actives if card["type"] == "neutral"]
        return blues, reds, neutrals

    def get_distribution(self):
        """Get the distribution over the classes depending on the difficulty"""

        cards_data_by_difficulty = {
            'easy': {"blue": 1, "red": 2, "neutral": 1, "none": 3},
            'medium': {"blue": 1, "red": 2, "neutral": 1, "none": 2},
            'default': {"blue": 0, "red": 3, "neutral": 1, "none": 2}
        }

        difficulty_level = self.board[0]["difficulty"]
        result = cards_data_by_difficulty.get(difficulty_level)
        return result

    def generate_computer_sequence(self):
        """Generate a sequence for the computer"""
        sequence = []
        card_type = None
        decay = 1

        run = card_type not in {"blue", "neutral"}

        while run:
            if len(self.blue) + len(self.red) + len(self.neutral) == 0:
                run = False

            weights = self._get_distribution_weights(decay, sequence)

            card_type = np.random.choice(["red", "blue", "neutral", "none"], p=weights)
            if card_type == "red":
                card_id = random.choice(self.red)
                self.red.remove(card_id)
                sequence.append(int(card_id))

            elif card_type == "blue":
                card_id = random.choice(self.blue)
                self.blue.remove(card_id)
                sequence.append(int(card_id))

            elif card_type == "neutral":
                card_id = random.choice(self.neutral)
                self.neutral.remove(card_id)
                sequence.append(int(card_id))

            else:
                run = False

            decay *= 0.35

        return sequence

    def _get_distribution_weights(self, decay, sequence):
        red_dist = self.distribution["red"] * decay if len(self.red) > 0 else 0
        blue_dist = self.distribution["blue"] if len(self.blue) > 0 else 0
        neutral_dist = self.distribution["neutral"] if len(self.neutral) > 0 else 0
        default_dist = self.distribution["none"] if len(sequence) != 0 else 0
        weights = [red_dist, blue_dist, neutral_dist, default_dist]
        return np.array(weights) / sum(weights)
