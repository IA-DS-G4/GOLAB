import tensorflow as tf
import keras
import numpy as np
from tensorflow.python.keras.layers import Input, Dense
from mcts import Action, Player
import typing
from typing import NamedTuple, Dict, List, Optional
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.models import Sequential, Model
import tensorflow_addons as tfa
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import (
    Dense,
    BatchNormalization,
    Reshape,
    Dropout,
    Flatten,
    Conv2D,
    MaxPool2D,
    Softmax,
    Activation,
    GlobalAveragePooling2D,
)


'''
Impementation of neural network in muzero algorithm for 7x7 game board

There are 4 networks:
- The Representation network (encoding network) --> convolutional network
- The Value network
- The Policy network
- The Reward network 

regularaization:
l2, dirichlet exploration noise 
'''


class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    policy_tensor: List[float]
    hidden_state: List[float]


class Network:

    def __init__(self, config):
        regularizer = l2(config.weight_decay)

        self.tot_training_steps = 0
        self.action_space_size = config.action_space_size

        self.representation = Sequential(
            [
                Dense(config.observation_space_size,
                      activation="relu",
                      input_shape=config.observation_space_size,
                      kernel_regularizer=regularizer),
                Conv2D(32, (3, 3),
                       activation='relu',
                       input_shape=config.observation_space_size,
                       kernel_regularizer=regularizer),
                MaxPool2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizer),
                MaxPool2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizer),
                Flatten(),
                Dense(64, activation="relu", kernel_regularizer=regularizer),
                Dense(config.hidden_layer_size, kernel_regularizer=regularizer)
            ]
        )

        self.representation = k.Sequential()
        self.representation.add(layers.Dense(config.observation_space_size, activation="relu", input_shape=config.observation_space_size,kernel_regularizer=regularizer))
        self.representation.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=config.observation_space_size, kernel_regularizer=regularizer))
        self.representation.add(layers.MaxPool2D((2, 2)))
        self.representation.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizer))
        self.representation.add(layers.MaxPool2D((2, 2)))
        self.representation.add(layers.Flatten())
        self.representation.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizer))
        self.representation.add(layers.Dense(config.hidden_layer_size, kernel_regularizer=regularizer))






        self.value = Sequential(
            [
                Dense(config.observation_space_size, activation="relu", input_shape=config.observation_space_size,
                      kernel_regularizer=regularizer),
                Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizer),
                MaxPool2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizer),
                MaxPool2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizer),
                Flatten(),
                Dense(64, activation="relu", kernel_regularizer=regularizer),
                Dense(config.hidden_layer_size, kernel_regularizer=regularizer)
            ]
        )

        self.policy = Sequential(
            [
                Dense(config.observation_space_size, activation="relu", input_shape=config.observation_space_size,
                      kernel_regularizer=regularizer),
                Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizer),
                MaxPool2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizer),
                MaxPool2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizer),
                Flatten(),
                Dense(64, activation="relu", kernel_regularizer=regularizer),
                Dense(config.hidden_layer_size, kernel_regularizer=regularizer),
                Softmax()
            ]
        )

        self.reward = Sequential(
            [
                Dense(config.observation_space_size, activation="relu", name="layer1", kernel_regularizer=regularizer),
                Dense(512, activation="relu", name="layer2", kernel_regularizer=regularizer),
                Dense(1024, activation="relu", name="layer3", kernel_regularizer=regularizer),
                Dense(512, activation="relu", name="layer4", kernel_regularizer=regularizer),
                Dense(256, activation="relu", name="layer5", kernel_regularizer=regularizer),
                Dense(1, name="layer6", kernel_regularizer=regularizer),
            ]
        )

        self.dynamics = Sequential(
            [
                Dense(config.observation_space_size, activation="relu", name="layer1", kernel_regularizer=regularizer),
                Dense(512, activation="relu", name="layer2", kernel_regularizer=regularizer),
                Dense(1024, activation="relu", name="layer3", kernel_regularizer=regularizer),
                Dense(512, activation="relu", name="layer4", kernel_regularizer=regularizer),
                Dense(config.hidden_layer_size, name="layer5", kernel_regularizer=regularizer),
            ]
        )




    def initial_inference(self, image):
        # representation + prediction function
        hidden_state = self.representation.predict(np.expand_dims(image, 0))
        # hidden_state = tf.keras.utils.normalize(hidden_state)

        value = self.value.predict(hidden_state)
        policy = self.policy.predict(hidden_state)
        reward = tf.constant([[0]], dtype=tf.float32)
        policy_p = policy[0]

        return NetworkOutput(value, reward, {Action(a): policy_p[a] for a in range(len(policy_p))}, policy, hidden_state)

    def recurrent_inference(self, hidden_state, action):
        # dynamics + prediction function

        a = hidden_state.numpy()[0]
        b = np.eye(self.action_space_size)[action.index]
        nn_input = np.concatenate((a, b))
        nn_input = np.expand_dims(nn_input, axis=0)

        next_hidden_state = self.dynamics.predict(nn_input)

        # next_hidden_state = tf.keras.utils.normalize(next_hidden_state)

        reward = self.reward.predict(nn_input)
        value = self.value.predict(next_hidden_state)
        policy = self.policy.predict(next_hidden_state)
        policy_p = policy[0]

        return NetworkOutput(value, reward, {Action(a): policy_p[a] for a in range(len(policy_p))}, policy, next_hidden_state)


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

    def training_steps(self):
        # How many steps / batches the network has been trained for.
        return self.tot_training_steps


class SharedStorage(object):

    def __init__(self, config):
        self.network = Network(config)

    def latest_network(self):
        return self.network

    def save_network(self, step: int, network: Network):
        pass