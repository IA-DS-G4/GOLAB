import tensorflow as tf
import keras
import numpy as np
from tensorflow.python.keras.layers import Input, Dense
from mcts import Action
import typing
from typing import NamedTuple, Dict, List, Optional

from tensorflow.python.keras.layers import (
    Dense,
    BatchNormalization,
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


class Network:

    def __init__(self, config):
        #regularizer = L2(config.weight_decay)
        self.tot_training_steps = 0
        self.action_space_size = config.action_space_size
        # Hyperparameters
        rnn_sizes = [64, 64]  # Sizes of LSTM layers
        head_hidden_sizes = [32, 16]  # Sizes of hidden layers in the heads
        normalize_hidden_state = True
        rnn_cell_type = 'lstm_norm'
        recurrent_activation = 'sigmoid'
        head_relu_before_norm = True
        nonlinear_to_hidden = True
        embed_actions = True



        self.representation = keras.Sequential(
                    [
                        Dense(config.observation_space_size, activation="relu", input_shape=config.observation_space_size),
                        Conv2D(32, (3, 3), activation='relu', input_shape=config.observation_space_size),
                        MaxPool2D((2, 2)),
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPool2D((2, 2)),
                        Conv2D(64, (3, 3), activation='relu'),
                        Flatten(),
                        Dense(64, activation="relu"),
                        Dense(config.hidden_layer_size)
                    ]
                )



        self.value = keras.Sequential(
                    [
                        Dense(config.observation_space_size, activation="relu", name="layer1"),
                        Dense(512, activation="relu", name="layer2"),
                        Dense(1024, activation="relu", name="layer3"),
                        Dense(512, activation="relu", name="layer4"),
                        Dense(256, activation="relu", name="layer5"),
                        Dense(1, name="layer6"),
                    ]
                )

        # #resnet + fully connected layers
        # #input size should be given in the form (x,y, channels) with channels 3 because resnet takes rgb images
        # model = ResNet50V2(include_top=False,input_tensor=None,input_shape=config.hidden_layer_size,classifier_activation="softmax")
        # x = model.output
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dense(512, activation='relu')(x)
        # preds = Dense(config.action_space_size, activation='softmax')(x)  # FC-layer
        # self.policy = Model(inputs=model.input, outputs=preds)

        self.policy  = keras.Sequential(
                    [
                        Dense(config.observation_space_size, activation="relu", name="layer1"),
                        Dense(512, activation="relu", name="layer2"),
                        Dense(1024, activation="relu", name="layer3"),
                        Dense(512, activation="relu", name="layer4"),
                        Dense(config.action_space_size, name="layer5", activation='softmax'),
                    ]
                )


        self.reward = keras.Sequential(
                    [
                        Dense(config.observation_space_size, activation="relu", name="layer1"),
                        Dense(512, activation="relu", name="layer2"),
                        Dense(1024, activation="relu", name="layer3"),
                        Dense(512, activation="relu", name="layer4"),
                        Dense(256, activation="relu", name="layer5"),
                        Dense(1, name="layer6"),
                    ]
                )

        #MLP
        self.dynamics = keras.Sequential(
                    [
                        Dense(config.observation_space_size, activation="relu", name="layer1"),
                        Dense(512, activation="relu", name="layer2"),
                        Dense(1024, activation="relu", name="layer3"),
                        Dense(512, activation="relu", name="layer4"),
                        Dense(config.hidden_layer_size, name="layer5"),
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