import copy 
import numpy as np


class GoBoard:
    def __init__(self, board_dimension, player, board_grid=[], game_history=[]):
        # Create a board with dimension board_dimension x board_dimension, player is the current player starting the game
        self.board_dimension = board_dimension
        self.player = player
        if len(game_history) > 0:
            self.board_grid = board_grid
            self.game_history = game_history
        else:
            self.board_grid = np.zeros((self.board_dimension,self.board_dimension), dtype="int32")
            self.game_history = []

    def flip_player(self):
        #Update the player to the other player
        self.player *= -1

    def add_move_to_history(self, r, c):
        #Add move (r, c) to the game_history field of the class
        self.game_history.append((self.player, r, c))

    def get_last_position(self):
        #Get the [r, c] position frmo the last moved,
        (player, r, c) = self.game_history[-1]
        return [r, c]

    def copy(self):
        # make a copy of the current board
        copy_board_grid = copy.deepcopy(self.board_grid)
        copy_game_history = copy.deepcopy(self.game_history)
        return GoBoard(self.board_dimension, self.player, copy_board_grid, copy_game_history)

    def __eq__(self, other):
      # Equality test to compare boards, games
        if isinstance(other, self.__class__) or issubclass(other.__class__, self.__class__) or issubclass(
                self.__class__, other.__class__):
            return self.board_dimension == other.board_dimension \
                and self.player == other.player \
                and self.board_grid == other.board_grid \
                and self.game_history == other.game_history
        return False

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def __str__(self):
        return str(self.board_dimension) + "x" + str(self.board_dimension) + " go board\n" \
            + "with current player " + str(self.player) + "\n with current grid" \
            + str(self.board_grid) + "\n with game history " + str(self.game_history)

