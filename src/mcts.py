import math
import numpy as np
from typing import List
from muzeroconfig import MuZeroConfig
from nn_models import Network, NetworkOutput
from Wrappers import ActionHistory, Action, Player, Node
import copy


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

    @staticmethod
    def play_game(config: MuZeroConfig, network: Network):
        game = config.new_game()

        while not game.done and len(game.board.game_history) < config.max_moves:
            # At the root of the search tree we use the representation function to
            # obtain a hidden state given the current observation.
            root = Node(0)
            current_observation = game.get_observation()
            MCTS.expand_node(root,
                        game.to_play(),
                        game.legal_actions(),
                        network.initial_inference(current_observation))
            MCTS.add_exploration_noise(config, root)

            # We then run a Monte Carlo Tree Search using only action sequences and the
            # model learned by the network.
            MCTS.run_mcts(config,root, game, network)
            action = MCTS.select_action(config, len(game.action_history), root, network)
            game.store_search_statistics(root)
            #print(f"move{action} from Player {game.board.player}")
            game.apply(action)
        print(f"simulated game! Winner is Player {game.utils.evaluate_winner(game.board.board_grid)}")
        return game

    # Core Monte Carlo Tree Search algorithm.
    # To decide on an action, we run N simulations, always starting at the root of
    # the search tree and traversing the tree according to the UCB formula until we
    # reach a leaf node.
    @staticmethod
    def run_mcts(config: MuZeroConfig,
                 root: Node,
                 game,
                 network: Network):

        min_max_stats = MinMaxStats()
        game_copy = copy.deepcopy(game)
        for _ in range(config.num_simulations):

            history = game_copy.get_action_history()
            node = root
            search_path = [node]

            while node.expanded():
                action, node = MCTS.select_child(config, node, min_max_stats)
                game_copy.apply(action)
                history.add_action(action)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
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
    @staticmethod
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
                      num_moves: int,
                      node: Node,
                      network: Network) -> Action:
        visit_counts = [(action,child.visit_count) for action, child in node.children.items()]
        t = config.visit_softmax_temperature_fn(num_moves=num_moves, training_steps=network.training_steps())
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

    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.
    @staticmethod
    def add_exploration_noise(config: MuZeroConfig, node: Node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
        frac = config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


class SharedStorage(object):

    def __init__(self, config):
        self.network = Network(config)

    def save_network(self,network):
        self.network = network

    def latest_network(self):
        return self.network



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
        return [(g.observation_list[i],
                 g.action_history[i:i + num_unroll_steps],
                 g.make_target(state_index=i,num_unroll_steps=num_unroll_steps, td_steps=td_steps, to_play=g.to_play(), action_space_size=action_space_size))
                for (g, i) in game_pos]

    def sample_game(self):
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[np.random.choice(range(len(self.buffer)))]

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return np.random.choice(range(len(game.rewards) - 1))

    def last_game(self):
        return self.buffer[-1]






