import math
import numpy as np
from typing import List
from muzeroconfig import MuZeroConfig
from nn_models import Network, NetworkOutput
from Wrappers import ActionHistory, Action, Player, Node
import copy


class MinMaxStats:
    """Class For storing MinMax values of Nodes"""

    def __init__(self):
        self.maximum =-float('inf')
        self.minimum = float('inf')
    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTS:
    """A class that runs Monte Carlo Tree Search. We use the UCB1 formula. to select a child and search from there"""

    @staticmethod
    def play_game(config: MuZeroConfig, network: Network):
        game = config.new_game()

        while not game.done and len(game.board.game_history) < config.max_moves:
            # At the root of the search tree we use the representation function to to get a hidden state from observation
            root = Node(0)
            current_observation = game.get_observation()
            # expand different board actions
            MCTS.expand_node(root,
                        game.to_play(),
                        game.legal_actions(),
                        network.initial_inference(current_observation))
            MCTS.add_exploration_noise(config, root)
            # running the MCTS using the network for evaluating positions and choosing actions
            MCTS.run_mcts(config,root, game, network)
            action = MCTS.select_action(config, root, network)
            game.store_search_statistics(root)
            game.apply(action)
        return game

    @staticmethod
    def run_mcts(config: MuZeroConfig,
                 root: Node,
                 game,
                 network: Network):
        # store minmax values
        min_max_stats = MinMaxStats()
        #copy game state to avoid changing the original game when running MCTS
        game_copy = copy.deepcopy(game)
        # we simulate the game for congig.num_simulations times
        for _ in range(config.num_simulations):

            history = game_copy.get_action_history()
            node = root
            search_path = [node]

            while node.expanded():
                action, node = MCTS.select_child(config, node, min_max_stats)
                game_copy.apply(action)
                history.add_action(action)
                search_path.append(node)
            # When traversing the tree we use the dynamics network to predict next states
            parent = search_path[-2]
            network_output = network.recurrent_inference(parent.hidden_state,
                                                         history.last_action())
            (MCTS.expand_node(node, history.to_play(), game_copy.legal_actions(), network_output))
            MCTS.backpropagate(search_path,
                          network_output.value,
                          history.to_play(),
                          config.discount,
                          min_max_stats)
            if game_copy.done:
                break
        del game_copy
    @staticmethod # take a softmax sample from a given distribution
    def softmax_sample(distribution, temperature: float):
        visit_counts = np.array([visit_counts for _ , visit_counts in distribution])
        actions = np.array([actions for actions , _ in distribution])
        visit_counts_exp = np.exp(visit_counts)
        policy = visit_counts_exp / np.sum(visit_counts_exp)
        policy = (policy ** (1 / temperature)) / (policy ** (1 / temperature)).sum()
        action_index = np.random.choice(actions, p=policy)

        return action_index
    @staticmethod
    def select_action(config,
                      node: Node,
                      network: Network) -> Action:
        visit_counts = [(action,child.visit_count) for action, child in node.children.items()]
        t = config.visit_softmax_temperature(training_steps=network.training_steps())
        action = Action(MCTS.softmax_sample(visit_counts, t))
        return action

    # Select the child with the highest UCB score.
    @staticmethod
    def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
        _, action, child = max(
            (MCTS.ucb_score(config, node, child, min_max_stats), action, child) for action, child in node.children.items())
        return action, child

    # The score for a node is based on its value, plus an exploration bonus based on the prior.
    @staticmethod
    def ucb_score(config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = min_max_stats.normalize(child.reward + config.discount * child.value())
        else:
            value_score = 0
        return prior_score + value_score

    # We expand a node using the value, reward and policy prediction obtained from the neural network.
    @staticmethod
    def expand_node(node: Node, to_play: Player, actions: List[Action], network_output: NetworkOutput):
        node.to_play = to_play
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward
        policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)

    # At the end of a simulation, we propagate the evaluation all the way up the tree to the root.
    @staticmethod
    def backpropagate(search_path: List[Node], value: float, to_play: Player, discount: float,
                      min_max_stats: MinMaxStats):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + discount * value


    @staticmethod # We add noise to the distribution to encourage exploration
    def add_exploration_noise(config: MuZeroConfig, node: Node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
        frac = config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


class NetworkStorage(object):
    def __init__(self, config):
        self.network = Network(config)

    def save_network(self,network):
        self.network = network

    def latest_network(self):
        return self.network



class GameStorage(object):
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.store = []

    def save_game(self, game):
        if len(self.store) > self.window_size:
            self.store.pop(0)
        self.store.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int, action_space_size: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.observation_list[i],
                 g.action_history[i:i + num_unroll_steps],
                 g.make_target(state_index=i,num_unroll_steps=num_unroll_steps, td_steps=td_steps))
                for (g, i) in game_pos]

    def sample_game(self):
        # get games from the gamestorage
        return self.store[np.random.choice(range(len(self.store)))]

    def sample_position(self, game) -> int:
        # sample a postion from the game
        return np.random.choice(range(len(game.rewards) - 1))

    def last_game(self):
        return self.store[-1]






