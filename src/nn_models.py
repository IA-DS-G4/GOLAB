import numpy as np
from Wrappers import Action
import typing
from typing import NamedTuple, Dict, List, Optional
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import tensorflow as tf
import tensorflow.python.keras as k


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

        self.representation = k.Sequential()
        self.representation.add(layers.Flatten(input_shape=config.observation_space_shape))
        self.representation.add(layers.Dense(config.hidden_layer_size, activation='relu'))
        self.representation.add(layers.Dense(config.hidden_layer_size, activation='relu'))

        self.value = k.Sequential()
        self.value.add(layers.Dense(config.hidden_layer_size, activation='relu'))
        self.value.add(layers.Dense(1, activation=None))  # No activation for value output

        self.policy = k.Sequential()
        self.policy.add(layers.Dense(config.hidden_layer_size, activation='relu'))
        self.policy.add(layers.Dense(config.action_space_size, activation=None))  # No activation for policy output

        self.reward = k.Sequential()
        self.reward.add(layers.Dense(config.hidden_layer_size, activation='relu'))
        self.reward.add(layers.Dense(1, activation=None))  # No activation for reward output

        self.dynamics = k.Sequential()
        self.dynamics.add(layers.Dense(config.hidden_layer_size, activation='relu'))
        self.dynamics.add(layers.Dense(config.hidden_layer_size, activation='relu'))

        self.tot_training_steps = 0

        self.action_space_size = config.action_space_size

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        hidden_state = self.representation(image)
        # hidden_state = tf.keras.utils.normalize(hidden_state)

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


    def save_model(self):

        representation_network = self.representation
        representation_network.save(r"../Saved models/representation_network")

        value_network = self.value
        value_network.save(r"../Saved models/value_network")

        dynamics_network = self.dynamics
        dynamics_network.save(r"../Saved models/dynamics_network")

        policy_network = self.policy
        policy_network.save(r"../Saved models/policy_network")

        reward_network = self.reward
        reward_network.save(r"../Saved models/reward_network")




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

