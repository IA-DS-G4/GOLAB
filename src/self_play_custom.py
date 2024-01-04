import time
import numpy as np
import math
from mcts import MinMaxStats, SharedStorage, ReplayBuffer
from muzeroconfig import MuZeroConfig
from nn_models import Network, NetworkOutput
from typing import List
from Wrappers import ActionHistory, Action, Player, Node


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.

def run_selfplay(config: MuZeroConfig,
                 storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    # while True: # in theory, this should be a job (i.e. thread) that runs continuously
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.

def play_game(config: MuZeroConfig, network: Network):
    game = config.new_game()

    while not game.is_finished() and len(game.board.game_history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.get_observation()
        expand_node(root,
                    game.to_play(),
                    game.legal_actions(),
                    network.initial_inference(current_observation))
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.action_history_list), root, network)
        game.apply(action)
        game.store_search_statistics(root)

    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.

def run_mcts(config: MuZeroConfig,
             root: Node,
             action_history: ActionHistory,
             network: Network):
    min_max_stats = MinMaxStats()

    for _ in range(config.num_simulations):

        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state,
                                                     history.last_action())
        expand_node(node, history.to_play(), history.action_space(), network_output)
        backpropagate(search_path,
                      network_output.value,
                      history.to_play(),
                      config.discount,
                      min_max_stats)


def softmax_sample(distribution, temperature: float):
    visit_counts = np.array([visit_counts for visit_counts, _ in distribution])
    visit_counts_exp = np.exp(visit_counts)
    policy = visit_counts_exp / np.sum(visit_counts_exp)
    policy = (policy ** (1 / temperature)) / (policy ** (1 / temperature)).sum()
    action_index = np.random.choice(range(len(policy)), p=policy)

    return action_index


def select_action(config: MuZeroConfig,
                  num_moves: int,
                  node: Node,
                  network: Network) -> Action:
    visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
    t = config.visit_softmax_temperature_fn(num_moves=num_moves, training_steps=network.training_steps())
    action = Action(softmax_sample(visit_counts, t))
    return action


# Select the child with the highest UCB score.

def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    _, action, child = max((ucb_score(config, node, child, min_max_stats), action, child) for action, child in node.children.items())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on the prior.

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

def expand_node(node: Node, to_play: Player, actions: List[Action], network_output: NetworkOutput):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    #policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    #policy_sum = sum(policy.values())
    #for action, p in policy.items():
    #    node.children[action] = Node(p / policy_sum)
    for action, p in network_output.policy_logits.items():
        node.children[action] = Node(p)


# At the end of a simulation, we propagate the evaluation all the way up the tree to the root.

def backpropagate(search_path: List[Node], value: float, to_play: Player, discount: float, min_max_stats: MinMaxStats):
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.

def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
