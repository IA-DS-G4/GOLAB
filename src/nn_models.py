import numpy as np
from Wrappers import Action
import typing
from typing import NamedTuple, Dict, List, Optional
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import tensorflow as tf
import tensorflow.python.keras as k
from keras import __version__
import copy
k.__version__ = __version__
from tensorflow.python.keras.regularizers import L2
from muzeroconfig import MuZeroConfig
from Go_7x7 import make_Go7x7_config


'''
Impementation of neural network in muzero algorithm for 7x7 game board

There are 4 networks:
- The Representation network 
- The Value network
- The Policy network
- The Reward network 
'''



class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    policy_tensor: List[float]
    hidden_state: List[float]


class Network(object):

    def __init__(self, config):
        self.hidden_layer_size = config.hidden_layer_size
        self.observation_space_shape = (1,) + config.observation_space_shape + (1,)
        regularizer = L2(config.weight_decay)


        #state encoder conv network
        self.representation = k.Sequential()
        self.representation.add(layers.Conv2D(64, (3, 3),activation='relu',kernel_regularizer=regularizer))
        self.representation.add(layers.Conv2D(64, (2, 2), activation='relu', kernel_regularizer=regularizer))
        self.representation.add(layers.Conv2D(64, (2, 2), activation='relu', kernel_regularizer=regularizer))
        self.representation.add(layers.Flatten())
        self.representation.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizer))
        self.representation.add(layers.Dropout(config.dropout_rate))
        self.representation.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizer))
        self.representation.add(layers.Dropout(config.dropout_rate))
        self.representation.add(layers.Dense(config.hidden_layer_size, kernel_regularizer=regularizer))

        #value network MLP
        self.value = k.Sequential()
        self.value.add(layers.Dense(config.hidden_layer_size, activation='relu', kernel_regularizer=regularizer))
        self.value.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizer))
        self.value.add(layers.Dropout(config.dropout_rate))
        self.value.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizer))
        self.value.add(layers.Dropout(config.dropout_rate))
        self.value.add(layers.Dense(256, activation="relu", kernel_regularizer=regularizer))
        self.value.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizer))
        self.value.add(layers.Dense(1,activation='relu', kernel_regularizer=regularizer))

        # policy network conv
        self.policy = k.Sequential()
        self.policy.add(layers.Dense(config.hidden_layer_size, activation='relu', kernel_regularizer=regularizer))
        self.policy.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizer))
        self.policy.add(layers.Dropout(config.dropout_rate))
        self.policy.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizer))
        self.policy.add(layers.Dropout(config.dropout_rate))
        self.policy.add(layers.Dense(256, activation="relu", kernel_regularizer=regularizer))
        self.policy.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizer))
        self.policy.add(layers.Dense(config.action_space_size, activation='softmax', kernel_regularizer=regularizer))

        #reward net MLP
        self.reward = k.Sequential()
        self.reward.add(layers.Dense(config.hidden_layer_size, activation='relu', kernel_regularizer=regularizer))
        self.reward.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizer))
        self.reward.add(layers.Dropout(config.dropout_rate))
        self.reward.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizer))
        self.reward.add(layers.Dropout(config.dropout_rate))
        self.reward.add(layers.Dense(256, activation="relu", kernel_regularizer=regularizer))
        self.reward.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizer))
        self.reward.add(layers.Dense(1,activation='relu', kernel_regularizer=regularizer))


        self.dynamics = k.Sequential()
        self.dynamics.add(layers.Dense(config.hidden_layer_size, activation='relu', kernel_regularizer=regularizer))
        self.dynamics.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizer))
        self.dynamics.add(layers.Dropout(0.2))
        self.dynamics.add(layers.Dense(1024, activation="relu", kernel_regularizer=regularizer))
        self.dynamics.add(layers.Dropout(0.2))
        self.dynamics.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizer))
        self.dynamics.add(layers.Dense(config.hidden_layer_size, kernel_regularizer=regularizer))


        self.tot_training_steps = 0
        self.backup_count = 0

        self.action_space_size = config.action_space_size

    def summarise(self):
        self.representation.build(self.observation_space_shape)
        self.value.build((0,100))
        self.policy.build(self.observation_space_shape)
        self.reward.build((0,100))
        self.dynamics.build((0,100))

        print(self.representation.summary(),
        self.value.summary(),
        self.policy.summary(),
        self.reward.summary(),
        self.dynamics.summary,
              )



    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        image = tf.expand_dims(image, axis= 3)
        image = tf.cast(image, dtype=tf.float32)
        hidden_state = self.representation(image)

        #check if tensor is non-zero, if so only then normalize to avoid nan's
        zero_check = tf.reduce_all(tf.equal(hidden_state, 0.0))
        if not zero_check:
            hidden_state = tf.linalg.normalize(hidden_state)[0]

        value = self.value(hidden_state)
        policy = self.policy(hidden_state)
        reward = tf.constant([[0]], dtype=tf.float32)
        policy_p = policy[0]
        return NetworkOutput(value,
                             reward,
                             {Action(a): policy_p[a] for a in range(len(policy_p))},
                             policy,
                             hidden_state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        a = hidden_state.numpy()[0]
        b = np.eye(self.action_space_size)[action.index]
        nn_input = np.concatenate((a, b))
        nn_input = np.expand_dims(nn_input, axis=0)

        next_hidden_state = self.dynamics(nn_input)

        # next_hidden_state = tf.keras.utils.normalize(next_hidden_state)

        reward = self.reward(nn_input)
        value = self.value(next_hidden_state)
        policy = self.policy(next_hidden_state)
        policy_p = policy[0]

        return NetworkOutput(value,
                             reward,
                             {Action(a): policy_p[a] for a in range(len(policy_p))},
                             policy,
                             next_hidden_state)

    def get_weights(self):
        # Returns the weights of this network.

        networks = (self.representation,
                    self.value,
                    self.policy,
                    self.dynamics,
                    self.reward)
        return [variables
                for variables_list in map(lambda n: n.weights, networks)
                for variables in variables_list]

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.tot_training_steps

    def save_network_deepcopy(self):

        self.representation.save(f"../Saved models/backup{self.backup_count}/representation_network")
        self.value.save(f"../Saved models/backup{self.backup_count}/value_network")
        self.dynamics.save(f"../Saved models/backup{self.backup_count}/dynamics_network")
        self.policy.save(f"../Saved models/backup{self.backup_count}/policy_network")
        self.reward.save(f"../Saved models/backup{self.backup_count}/reward_network")


if __name__ == "__main__":
    network =Network(make_Go7x7_config())
    network.summarise()