import tensorflow as tf
import keras
import numpy as np
from tensorflow.python.keras.layers import Input, Dense
from mcts import Action, Player
from typing import NamedTuple, Dict, List, Optional
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
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


        #ResNet50V2
        self.representation = keras.Sequential(
                    [
                        Dense(config.observation_space_size, activation="relu", name="layer1"),
                        Dense(512, activation="relu", name="layer2"),
                        Dense(1024, activation="relu", name="layer3"),
                        Dense(512, activation="relu", name="layer4"),
                        Dense(config.hidden_layer_size, name="layer5"),
                    ]
                )


        #ResNet50V2 + fully connected layers
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

        #resnet
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


        self.tot_training_steps = 0
        self.action_space_size = config.action_space_size

    def initial_inference(self, image):
        # representation + prediction function
        hidden_state = self.representation(np.expand_dims(image, 0))
        # hidden_state = tf.keras.utils.normalize(hidden_state)

        value = self.value(hidden_state)
        policy = self.policy(hidden_state)
        reward = tf.constant([[0]], dtype=tf.float32)
        policy_p = policy[0]

        return NetworkOutput(value, reward, {Action(a): policy_p[a] for a in range(len(policy_p))}, policy, hidden_state)

    def recurrent_inference(self, hidden_state, action):
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

        return NetworkOutput(value, reward, {Action(a): policy_p[a] for a in range(len(policy_p))}, policy, next_hidden_state)

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