import copy
import json
import math
import pathlib
import pickle
import sys
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import go_muzero_config
import diagnose_model
from model import MuZeroNetwork
import replay_buffer
import self_play
import shared_storage
import trainer


class MuZero:
    def __init__(self):
        self.Game = go_muzero_config.MuzeroGame()
        self.config = go_muzero_config.MuZeroConfig()

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)


        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }

        self.replay_buffer = {}

        cpu_weights = get_initial_weights(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(cpu_weights)

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        return

    def logging_loop(self, num_gpus): # Keep track of the training performance.
        return


    def test(self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0):


    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
               Load a model and/or a saved replay buffer.

               Args:
                   checkpoint_path (str): Path to model.checkpoint or model.weights.

                   replay_buffer_path (str): Path to replay_buffer.pkl
               """
        # Load checkpoint
        if checkpoint_path:
            checkpoint_path = pathlib.Path(checkpoint_path)
            self.checkpoint = torch.load(checkpoint_path)
            print(f"\nUsing checkpoint from {checkpoint_path}")

        # Load replay buffer
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_played_steps"] = replay_buffer_infos[
                "num_played_steps"
            ]
            self.checkpoint["num_played_games"] = replay_buffer_infos[
                "num_played_games"
            ]
            self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                "num_reanalysed_games"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")
        else:
            print(f"Using empty buffer.")
            self.replay_buffer = {}
            self.checkpoint["training_step"] = 0
            self.checkpoint["num_played_steps"] = 0
            self.checkpoint["num_played_games"] = 0
            self.checkpoint["num_reanalysed_games"] = 0

    # ... (unchanged)


def get_initial_weights(self, config):
    model = model.MuZeroNetwork(config)
    weights = model.get_weights()
    summary = str(model).replace("\n", " \n\n")
    return weights, summary


#Search for hyperparameters
def hyperparameter_search(game_name, parametrization, budget, parallel_experiments, num_tests):
    pass


def load_model_menu(muzero, game_name):
    pass


if __name__ == "__main__":
    #initialise Example_code
    muzero = MuZero()
