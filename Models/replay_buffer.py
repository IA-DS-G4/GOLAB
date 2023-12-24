import copy
import time
import numpy as np
import torch
import models


#the replaybuffer saves all self played games in a useable way so the model can use it for training
class ReplayBuffer:
    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum([len(game_history.root_values) for game_history in self.buffer.values()])
        if self.total_samples != 0:
            print(f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n")

        np.random.seed(self.config.seed)

    def save_game(self, game_history, shared_storage=None):
        if self.config.PER:
            if game_history.priorities is not None:
                game_history.priorities = np.copy(game_history.priorities)
            else:
                priorities = [
                    np.abs(root_value - self.compute_target_value(game_history, i)) ** self.config.PER_alpha
                    for i, root_value in enumerate(game_history.root_values)
                ]

                game_history.priorities = np.array(priorities, dtype="float32")
                game_history.game_priority = np.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage["num_played_games"] = self.num_played_games
            shared_storage["num_played_steps"] = self.num_played_steps

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        for game_id, game_history, game_prob in self.sample_n_games(self.config.batch_size):
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(game_history, game_pos)

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                )
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [min(self.config.num_unroll_steps, len(game_history.action_history) - game_pos)] * len(actions)
            )
            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            weight_batch = np.array(weight_batch, dtype="float32") / max(weight_batch)

        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def sample_game(self, force_uniform=False):
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = np.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= np.sum(game_probs)
            game_index = np.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs
