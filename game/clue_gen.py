import pickle
from itertools import chain

import numpy as np
from numba import njit
from scipy import spatial


@njit(fastmath=True)
def binary_search(arr):
    mini, mid, maxi = 0, 0, arr.shape[0]
    rand = arr[-1] * np.random.random()
    while mini < maxi:
        mid = mini + ((maxi - mini) >> 1)
        if rand > arr[mid]:
            mini = mid + 1
        else:
            maxi = mid
    return mini


class ClueGenerator:
    def __init__(self, relevant_words, relevant_vectors, board, false_guesses, threshold=0.4, runs=10000):
        self.relevant_words = relevant_words
        self.relevant_vectors = relevant_vectors
        self.board = board
        self.false_guesses = false_guesses
        self.threshold = threshold
        self.runs = runs
        self.passive_words = None
        self.words = None
        self.blue_cards = None
        self.red_cards = None
        self.neutral_cards = None
        self.true_guesses = None

    @staticmethod
    @njit(fastmath=True)
    def _compute_exp_score(similarities, n_blue, trials):
        expected_score = 0

        for _ in range(trials):
            trial_score = 0
            cumsum = np.cumsum(similarities)
            run = True
            while run:
                sample = binary_search(cumsum)
                if sample < n_blue:
                    if sample == 0:
                        cumsum[sample] = 0
                    else:
                        difference = cumsum[sample] - cumsum[sample - 1]
                        cumsum[sample:] -= difference
                    trial_score += 1
                else:
                    run = False
            expected_score += trial_score
        expected_score /= trials
        return expected_score

    def _collect_card_types(self):
        not_actives = [card for card in self.board if not card["active"]]
        blues = [c["name"].replace(" ", "") for c in not_actives if c["type"] == "blue"]
        reds = [c["name"].replace(" ", "") for c in not_actives if c["type"] == "red"]
        neutrals = [c["name"].replace(" ", "") for c in not_actives if c["type"] == "neutral"]
        assasin = [c["name"].replace(" ", "") for c in not_actives if c["type"] == "assassin"][0]
        return blues, reds, neutrals, assasin

    def _colelct_right_guesses(self):
        with open(self.relevant_words, 'rb') as f:
            relevant_words = pickle.load(f)
        right_blues = chain.from_iterable(relevant_words[w] for w in self.blue_cards)
        return set(right_blues).difference(self.false_guesses)

    def _collect_relevant_vectors(self):
        with open(self.relevant_vectors, 'rb') as f:
            relevant_vectors = pickle.load(f)
        return relevant_vectors

    def _prepare(self):
        self.relevant_vectors = self._collect_relevant_vectors()
        self.words = [card["name"].replace(" ", "") for card in self.board]
        self.blue_cards, self.red_cards, self.neutral_cards, self.assassin_card = self._collect_card_types()
        self.bad_words = self.red_cards + [self.assassin_card] + self.neutral_cards
        self.blue_vectors = np.array([self.relevant_vectors[w] for w in self.blue_cards], dtype=np.float32)
        self.bad_vectors = np.array([self.relevant_vectors[w] for w in self.bad_words], dtype=np.float32)
        self.true_guesses = self._colelct_right_guesses()

    def _compute_guess_score(self, guess):
        guess_vector = self.relevant_vectors[guess]

        blue_sim_dists = [1 - spatial.distance.cosine(guess_vector, v) for v in self.blue_vectors]
        bad_sim_dists = [1 - spatial.distance.cosine(guess_vector, v) for v in self.bad_vectors]
        blue_sims = np.array(blue_sim_dists, dtype=np.float32)
        bad_sims = np.array(bad_sim_dists, dtype=np.float32)

        best_blue_sims = blue_sims[blue_sims > self.threshold]
        best_bad_sims = bad_sims[bad_sims > self.threshold]
        best_sims = np.concatenate([best_blue_sims, best_bad_sims])

        if len(best_bad_sims) == 0:
            return (len(best_blue_sims), np.sum(best_blue_sims)), guess
        elif len(best_blue_sims) == 0:
            return (0, 0), guess
        else:
            return (self._compute_exp_score(best_sims, len(best_blue_sims), self.runs), 0), guess

    def _collect_targets(self, guess, clue_score):
        blue_similarities = np.array([
            1 - spatial.distance.cosine(self.relevant_vectors[guess], self.relevant_vectors[w])
            for w in self.blue_cards
        ])
        best_blue = set(np.array(self.blue_cards)[np.argsort(-blue_similarities)][:clue_score])
        return [
            card['id']
            for card in self.board
            if card['name'].replace(' ', '') in best_blue
        ]

    def execute(self):
        self._prepare()
        guess_scores = [self._compute_guess_score(g) for g in self.true_guesses]
        score, clue = max(guess_scores, key=lambda gs: gs[0])
        targets = self._collect_targets(clue, int(score[0]))
        return clue, int(score[0]), targets
