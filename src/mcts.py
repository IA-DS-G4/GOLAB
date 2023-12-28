import math
import numpy as np
from typing import Dict, List, Optional
from Go_7x7 import MuZeroConfig
from nn_models import Network, NetworkOutput


MAXIMUM_FLOAT_VALUE = float('inf')

class Action(object):

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

    def __str__(self):
        return str(self.index)


class Player(object):

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

    def __str__(self):
        return str(self.index)

class ActionHistory(object):
    """Simple history container used inside the search.
       Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player(1)

class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self):
        self.maximum =-MAXIMUM_FLOAT_VALUE
        self.minimum = MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class Node(object):

    def __init__(self, prior: float):

        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:

        return len(self.children) > 0

    def value(self) -> float:

        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count

class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run_mcts(self,
                 root: Node,
                 action_history: ActionHistory,
                 network: Network):

        min_max_stats = MinMaxStats()

        for _ in range(self.config.num_simulations):
            history = action_history.clone()
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node, min_max_stats)
                history.add_action(action)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            network_output = network.recurrent_inference(parent.hidden_state,
                                                         history.last_action())
            self.expand_node(node, history.to_play(), history.action_space(), network_output)

            self.backpropagate(search_path,
                          network_output.value,
                          history.to_play(),
                          self.config.discount,
                          min_max_stats)


        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
    @staticmethod
    def softmax_sample(distribution, temperature: float):

        visit_counts = np.array([visit_counts for visit_counts, _ in distribution])
        visit_counts_exp = np.exp(visit_counts)
        policy = visit_counts_exp / np.sum(visit_counts_exp)
        policy = (policy ** (1 / temperature)) / (policy ** (1 / temperature)).sum()
        action_index = np.random.choice(range(len(policy)), p=policy)

        return action_index


    def select_action(self,
                      num_moves: int,
                      node: Node,
                      network: Network):

        visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
        t = self.config.visit_softmax_temperature_fn(num_moves=num_moves, training_steps=network.training_steps())
        action = self.softmax_sample(visit_counts, t)
        return action

    def select_child(self, node: Node, min_max_stats: MinMaxStats):

        _, action, child = max(
            (self.ucb_score(node, child, min_max_stats), action, child) for action, child in node.children.items())
        return action, child

    def ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:

        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = min_max_stats.normalize(child.reward + self.config.discount * child.value())
        else:
            value_score = 0

        return prior_score + value_score

    def expand_node(self,node: Node, to_play: Player, actions: List[Action], network_output: NetworkOutput):

        node.to_play = to_play
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward
        # policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
        # policy_sum = sum(policy.values())
        # for action, p in policy.items():
        #    node.children[action] = Node(p / policy_sum)
        for action, p in network_output.policy_logits.items():
            node.children[action] = Node(p)

    def backpropagate(self,search_path: List[Node], value: float, to_play: Player, discount: float,
                      min_max_stats: MinMaxStats):

        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + discount * value

    def add_exploration_noise(self, node: Node):

        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def play_game(self, network: Network):
        game = self.config.new_game()

        while not game.terminal() and len(game.history) < self.config.max_moves:
            # At the root of the search tree we use the representation function to
            # obtain a hidden state given the current observation.
            root = Node(0)
            current_observation = game.make_image(-1)
            self.expand_node(root,
                        game.to_play(),
                        game.legal_actions(),
                        network.initial_inference(current_observation))
            self.add_exploration_noise(root)

            # We then run a Monte Carlo Tree Search using only action sequences and the
            # model learned by the network.
            self.run_mcts(root, game.action_history(), network)
            action = self.select_action(len(game.history), root, network)
            game.apply(action)
            game.store_search_statistics(root)

        return game







