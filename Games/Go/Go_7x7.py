import datetime
import math
import pathlib
import numpy
import torch
from go_board import GoBoard
from go_utils import GoUtils
from abstract_game import AbstractGame


class MuZeroConfig:

    def __init__(self,
                 action_space_size: int,
                 observation_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 training_episodes: int,
                 hidden_layer_size: int,
                 visit_softmax_temperature_fn,
                 known_bounds: Optional[KnownBounds] = None):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (3, 7, 7)  # Dimensions of the game observation, must be 3 (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(-1,(7 * 7)))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        ### Self-Play
        self.action_space_size = len(self.action_space)
        self.observation_space_size = 7*7
        self.num_actors = 2

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(500e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1000)
        self.batch_size = batch_size
        self.num_unroll_steps = 500
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.training_episodes = training_episodes

        self.hidden_layer_size = hidden_layer_size

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

    def new_game(self):
        return Go7x7


class Go7x7config(MuZeroConfig):
    def new_game(self):
        return Go7x7(self.action_space_size, self.discount)
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

    return Go7x7config(action_space_size=2,
                          observation_space_size=4,
                          max_moves=500,
                          discount=0.997,
                          dirichlet_alpha=0.25,
                          num_simulations=150,
                          batch_size=100,
                          td_steps=7,
                          num_actors=1,
                          lr_init=0.0001,
                          lr_decay_steps=5000,
                          training_episodes=225,
                          hidden_layer_size=32,
                          visit_softmax_temperature_fn=visit_softmax_temperature)


class Go7x7:
    def __init__(self):
        self.board_size = 7
        self.player = 1 # Black goes first
        self.board = GoBoard(board_dimension=self.board_size, player=self.player)
        self.utils = GoUtils()

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

    def render(self):
        pass

    def human_input_to_action(self):
        pass
        return 1

    def action_to_human_input(self, action):
        pass
        return 1

