import tensorflow as tf
import keras
import numpy as np
from Wrappers import Action
import typing
from typing import NamedTuple, Dict, List, Optional

from tensorflow.python.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPool2D,
    Activation,
    GlobalAveragePooling2D,
)


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
        # regularizer = L2(config.weight_decay)

        self.representation = keras.Sequential([Dense(config.hidden_layer_size, activation='relu'),
                                                Dense(config.hidden_layer_size)])

        self.value = keras.Sequential([Dense(config.hidden_layer_size, activation='relu'),
                                       Dense(1, activation='relu')])

        self.policy = keras.Sequential([Dense(config.hidden_layer_size, activation='relu'),
                                        Dense(config.action_space_size, activation='softmax')])

        self.reward = keras.Sequential([Dense(config.hidden_layer_size, activation='relu'),
                                        Dense(1, activation='relu')])

        self.dynamics = keras.Sequential([Dense(config.hidden_layer_size, activation='relu'),
                                          Dense(config.hidden_layer_size)])

        self.tot_training_steps = 0

        self.action_space_size = config.action_space_size

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function

        hidden_state = self.representation(np.expand_dims(image, 0))

        # hidden_state = tf.keras.utils.normalize(hidden_state)

        value = self.value(hidden_state)
        policy = self.policy(hidden_state)
        reward = tf.constant([[0]], dtype=tf.float32)
        policy_p = policy[0]
        list_actions = {Action(a): policy_p[a] for a in range(len(policy_p))}
        print(list_actions)
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

    def get_weights_func(self):
        # Returns the weights of this network.

        def get_variables():
            networks = (self.representation,
                        self.value,
                        self.policy,
                        self.dynamics,
                        self.reward)
            return [variables
                    for variables_list in map(lambda n: n.weights, networks)
                    for variables in variables_list]

        return get_variables

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


class SharedStorage(object):

    def __init__(self, config):
        self.network = Network(config)

    def latest_network(self):
        return self.network

    def save_network(self, step: int, network: Network):
        pass