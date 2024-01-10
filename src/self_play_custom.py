from mcts import SharedStorage, ReplayBuffer, MCTS
from muzeroconfig import MuZeroConfig


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.

def run_selfplay(config: MuZeroConfig,
                 storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    mcts = MCTS(config)
    for _ in range(config.batch_size):
        network = storage.latest_network()
        game = mcts.play_game(network)
        replay_buffer.save_game(game)

# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.

