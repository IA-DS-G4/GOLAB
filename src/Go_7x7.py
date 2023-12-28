import numpy
from go_board import GoBoard
from go_utils import GoUtils
from typing import Dict, List, Optional
from nn_models import Network
from muzeroconfig import MuZeroConfig


def make_Go7x7_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):

        # higher temperature higher exploration

        if training_steps < 100:
            return 3
        elif training_steps < 125:
            return 2
        elif training_steps < 150:
            return 1
        elif training_steps < 175:
            return 0.5
        elif training_steps < 200:
            return 0.250
        elif training_steps < 225:
            return 0.125
        elif training_steps < 250:
            return 0.075
        else:
            return 0.001

    return MuZeroConfig(action_space_size=2,
                          observation_space_size=4,
                          max_moves=500,
                          discount=0.997,
                          dirichlet_alpha=0.25,
                          num_simulations=150,
                          batch_size=100,
                          td_steps=7,
                          lr_init=0.0001,
                          lr_decay_steps=5000,
                          training_episodes=225,
                          hidden_layer_size=32,
                          visit_softmax_temperature_fn=visit_softmax_temperature,
                          Game= Go7x7)


class Go7x7:
    def __init__(self):
        self.board_size = 7
        self.player = 1 # Black goes first
        self.board = GoBoard(board_dimension=self.board_size, player=self.player)
        self.utils = GoUtils()
        self.observation_space_shape = (3,7,7)
        self.observation_space_size = 7*7
        self.action_space = list(range(-1,(7*7)))

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.player = 1
        self.board = GoBoard(board_dimension=self.board_size, player=self.player)
        return self.get_observation()

    def step(self, action):
        r = numpy.floor(action / self.board_size)
        c = action % self.board_size
        move = (r,c)
        if action == -1:
            move = (-1,-1)
        self.utils.make_move(board=self.board,move=move)
        done = self.utils.is_game_finished(board=self.board)
        if done:
            reward = 1 if self.utils.evaluate_winner(board_grid=self.board.board_grid)[0] == self.player else -1
        else:
            reward = 0
        self.player *= -1
        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board.board_grid == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board.board_grid == -1, 1.0, 0.0)
        board_to_play = numpy.full((self.board_size,self.board_size), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        # Pass = -1 is a valid move
        legal = [-1]
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.utils.is_valid_move(board=self.board,move=(i,j)):
                    legal.append(i * self.board_size + j)
        return legal

    def is_finished(self):
        return self.utils.is_game_finished(board=self.board)






