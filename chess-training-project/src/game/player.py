# FILE: /chess-training-project/chess-training-project/src/game/player.py

class Player:
    def __init__(self, name):
        self.name = name

    def make_move(self, board, move):
        """Make a move on the given board."""
        if board.is_valid_move(move):
            board.apply_move(move)
            return True
        return False

    def get_move(self):
        """Get the player's move. This method should be overridden by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")