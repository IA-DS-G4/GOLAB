import math
import numpy as np
from typing import List
from muzeroconfig import MuZeroConfig
from nn_models import Network, NetworkOutput
from Wrappers import ActionHistory, Action, Player, Node


MAXIMUM_FLOAT_VALUE = float('inf')


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


class SharedStorage(object):

    def __init__(self, config):
        self.network = Network(config)

    def latest_network(self):
        return self.network

    def save_network(self):
        self.network.save_model()
        pass


class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int, action_space_size: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i),
                 g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play(), action_space_size))
                for (g, i) in game_pos]

    def sample_game(self):
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[np.random.choice(range(len(self.buffer)))]

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return np.random.choice(range(len(game.rewards) - 1))

    def last_game(self):
        return self.buffer[-1]


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


    def action_history(self) -> ActionHistory:

        return ActionHistory(self.history, self.action_space_size)






