import json
import os

from flask import Flask, render_template, request, jsonify

from game.board import Board
from game.clue_gen import ClueGenerator
from game.comp import Computer

app = Flask(__name__)

FILES = {
    'words': os.path.abspath(os.path.join('data', 'codenames_words')),
    'valid_words': os.path.abspath(os.path.join('data', 'relevant_words')),
    'valid_word_vectors': os.path.abspath(os.path.join('data', 'relevant_vectors')),
}


@app.route('/', methods=["POST", "GET"])
def index():
    """Homepage for the website. Create a random board."""
    board = Board(FILES['words']).board
    initial_cell = {"difficulty": "easy", "invalid_guesses": []}
    board.insert(0, initial_cell)
    return render_template('page.html', board=board)


@app.route("/update", methods=["POST"])
def update_page():
    """Update the page with the details from the current board"""
    return render_template('page.html', board=json.loads(request.data))


@app.route("/computer_turn", methods=["POST"])
def computer_turn():
    """Get a series of computer moves"""
    board = json.loads(request.data)
    sequence = Computer(board).generate_computer_sequence()
    return jsonify(sequence=sequence)


@app.route("/clue", methods=["POST"])
def clue():
    """Generate a clue"""
    board = json.loads(request.data)
    predictor = ClueGenerator(
        relevant_words=FILES['valid_words'],
        relevant_vectors=FILES['valid_word_vectors'],
        board=board[1:],
        false_guesses=set(board[0]['invalid_guesses']),
        threshold=0.45
    )
    _clue, clue_score, targets = predictor.execute()
    return jsonify(clue=_clue, targets=targets)


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='127.0.0.1', port=8080, debug=True)
