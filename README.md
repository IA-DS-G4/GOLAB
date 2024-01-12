
# MuZero Algorithm Implementation for Go

## Introduction
This report details the implementation of our MuZero algorithm for the board game Go. The implementation focuses on 7x7 and 9x9 board sizes, adapting the MuZero algorithm based on the pseudocode from DeepMind. We chose MuZero over Alphazero because it can generalize across a broader range of environments, especially in scenarios with incomplete information or more complex, dynamic rules.

## Go Game Implementation
We structured the Go implementation into 3 scripts: The board, The Utilities, used to act on the board and extract information, and the Graphical interface implementation.
You can run the `play_human_human.py` script to play with another human or the `play_Muzero_interactive` script to play against MuZero.

## Monte Carlo Tree Search
For the MCTS, we first tried to teach MuZero the rules of the game by punishing it with a negative reward when doing an illegal move. We noticed the training took very long for our network to even learn the simple rules, so we adjusted the approach to only allow legal moves. This resulted in large memory and computation overhead because we used our game implementation to calculate legal moves. In the future, we should think about the memory and performance of the game to avoid problems like that.

## Network Architecture
For the core of the model, we decided to use the following neural network architectures. A convolutional neural network (CNN) was employed to encode the game state into a hidden state vector. We chose a CNN to capture the spatial features and relationships of the go game board. Policy, Value, Reward, and Dynamics networks were implemented using fully connected multilayer perceptrons. As input, they take the hidden state vector. The value network estimates the probability of winning from the current state, the reward network evaluates the immediate reward from an action, and the dynamics network predicts the next state and immediate reward while the policy network is used to predict the next move.

## Training
We had very limited computational resources and training time, as we were a group of two and training was done on a chromebook, so both models used small batch sizes of 16 games per batch for each training step. To optimize the training we stored all MCTS rollouts and scores for later training. The 7x7 model was trained for 90 episodes and the 9x9 model was trained for 40 episodes.
