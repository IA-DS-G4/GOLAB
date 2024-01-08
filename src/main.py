import time
import numpy as np
import tensorflow as tf
import keras
from muzeroconfig import MuZeroConfig
from mcts import MinMaxStats, SharedStorage, ReplayBuffer
from matplotlib import pyplot as plt
from nn_models import Network
from self_play_custom import run_selfplay
from Wrappers import Action, ActionHistory
from IPython.display import clear_output
from Go_7x7 import make_Go7x7_config

def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return tf.losses.mean_squared_error(target, prediction)


def scale_gradient(tensor, scale: float):
    # Scales the gradient for the backward pass.
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch, weight_decay: float):
    with tf.GradientTape() as tape:

        loss = 0

        for image, actions, targets in batch:

            # Initial step, from the real observation.
            value, reward, _, policy_t, hidden_state = network.initial_inference(image)
            predictions = [(1.0, value, reward, policy_t)]

            # Recurrent steps, from action and previous hidden state.
            for action in actions:
                value, reward, _, policy_t, hidden_state = network.recurrent_inference(hidden_state, Action(action))
                predictions.append((1.0 / len(actions), value, reward, policy_t))

                hidden_state = scale_gradient(hidden_state, 0.5)

            for k, (prediction, target) in enumerate(zip(predictions, targets)):

                gradient_scale, value, reward, policy_t = prediction
                target_value, target_reward, target_policy = target

                l_a = scalar_loss(value, [target_value])

                if k > 0:
                    l_b = tf.dtypes.cast(scalar_loss(reward, [target_reward]), tf.float32)
                else:
                    l_b = 0

                if target_policy == []:
                    l_c = 0
                else:
                    # l_c = tf.nn.softmax_cross_entropy_with_logits(logits=policy_t, labels=target_policy)
                    cce = keras.losses.CategoricalCrossentropy()
                    l_c = cce([target_policy], policy_t)

                l = l_a + l_b + l_c

                loss += scale_gradient(l, gradient_scale)

        loss /= len(batch)

        for weights in network.get_weights():
            loss += weight_decay * tf.nn.l2_loss(weights)

    # optimizer.minimize(loss) # this is old Tensorflow API, we use GradientTape

    gradients = tape.gradient(loss, [network.representation.trainable_variables,
                                     network.dynamics.trainable_variables,
                                     network.policy.trainable_variables,
                                     network.value.trainable_variables,
                                     network.reward.trainable_variables])

    optimizer.apply_gradients(zip(gradients[0], network.representation.trainable_variables))
    optimizer.apply_gradients(zip(gradients[1], network.dynamics.trainable_variables))
    optimizer.apply_gradients(zip(gradients[2], network.policy.trainable_variables))
    optimizer.apply_gradients(zip(gradients[3], network.value.trainable_variables))
    optimizer.apply_gradients(zip(gradients[4], network.reward.trainable_variables))

    return loss


def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, iterations: int):
    network = storage.latest_network()
    learning_rate = config.lr_init * config.lr_decay_rate ** (iterations / config.lr_decay_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.action_space_size)
    loss = update_weights(optimizer, network, batch, config.weight_decay)

    network.tot_training_steps += 1

    return loss

def launch_job(f, *args):
    f(*args)


def muzero(config: MuZeroConfig):
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    rewards = []
    losses = []
    moving_averages = []

    t = time.time()

    for i in range(config.training_episodes):
        #save model every 25 episodes
        if i % 25 == 0:
            storage.save_network()

        # self-play
        launch_job(run_selfplay, config, storage, replay_buffer)

        # print and plot rewards
        game = replay_buffer.last_game()
        reward_e = game.total_rewards()
        rewards.append(reward_e)
        moving_averages.append(np.mean(rewards[-20:]))

        for _ in range(10):
            clear_output(wait=True)

        print('Episode ' + str(i + 1) + ' ' + 'reward: ' + str(reward_e))
        print('Moving Average (20): ' + str(np.mean(rewards[-20:])))
        print('Moving Average (100): ' + str(np.mean(rewards[-100:])))
        print('Moving Average: ' + str(np.mean(rewards)))
        print('Elapsed time: ' + str((time.time() - t) / 60) + ' minutes')

        plt.plot(rewards)
        plt.plot(moving_averages)
        plt.show()

        # training
        loss = train_network(config, storage, replay_buffer, i)

        # print and plot loss
        print('Loss: ' + str(loss))
        losses.append(loss)
        plt.plot(losses)
        plt.show()

if __name__ == "__main__":
    muzero(make_Go7x7_config())

