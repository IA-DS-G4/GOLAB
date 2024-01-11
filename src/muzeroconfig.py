import typing
from typing import Optional, List
import collections
from Wrappers import Player, Action, ActionHistory, Node


KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MuZeroConfig(object):

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
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 training_episodes: int,
                 hidden_layer_size: int,
                 model_name: str,
                 visit_softmax_temperature_fn,
                 known_bounds: Optional[KnownBounds] = None,
                 ):
        ### Self-Play
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size
        self.observation_space_shape = observation_space_shape
        self.num_actors = num_actors

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
        self.window_size = batch_size
        self.batch_size = batch_size
        self.num_unroll_steps = 3
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.training_episodes = training_episodes
        self.hidden_layer_size = hidden_layer_size

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = 0.1

        self.model_name = model_name

    def new_game(self):
        return Game(self.action_space_size, self.discount)


class Game(object):
    def __init__(self, action_space_size: int, discount: float):
        self.player = 1
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

        return Player(self.player)

    def action_history(self) -> ActionHistory:

        return ActionHistory(self.history, self.action_space_size, self.to_play())

    def total_rewards(self):

        return sum(self.rewards)



