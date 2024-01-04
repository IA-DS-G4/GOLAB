import typing
from typing import Optional, List
import collections
from Wrappers import Player, Action, ActionHistory, Node


KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MuZeroConfig:

    def __init__(self,
                 action_space_size: int,
                 observation_space_size: int,
                 observation_space_shape: (int,int),
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 training_episodes: int,
                 hidden_layer_size: int,
                 visit_softmax_temperature_fn,
                 known_bounds: Optional[KnownBounds] = None,
                 Game = None):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_space_size = observation_space_size
        self.observation_space_shape = observation_space_shape  # Dimensions of the game observation, must be 3 (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space_size = action_space_size  # Fixed list of all possible actions. You should only edit the length

        ### Self-Play
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

        #rewards in the environment are -1 for loose and 1 for win
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
        return Game(self.action_space_size, self.discount)


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size: int, discount: float):

        self.environment = self.create_environment()
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

    def create_environment(self):

        # Game specific environment.
        pass

    def terminal(self) -> bool:

        # Game specific termination rules.
        pass

    def legal_actions(self) -> List[Action]:

        # Game specific calculation of legal actions.
        pass

    def apply(self, action: Action):

        reward = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):

        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index: int):

        # Game specific feature planes.
        pass

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player,
                    action_space_size: int):

        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i  # pytype: disable=unsupported-operands

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = None

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, last_reward, []))

        return targets

    def to_play(self) -> Player:

        return Player(1)

    def action_history(self) -> ActionHistory:

        return ActionHistory(self.history, self.action_space_size)

    def total_rewards(self):

        return sum(self.rewards)



