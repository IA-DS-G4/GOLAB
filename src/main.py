import tensorflow as tf
import keras
from muzeroconfig import MuZeroConfig
from mcts import SharedStorage, ReplayBuffer, MCTS
from matplotlib import pyplot as plt
from nn_models import Network
from Wrappers import Action
from Go_7x7 import make_Go7x7_config
from Go_9x9 import make_Go9x9_config
from tqdm import tqdm


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("TensorFlow is using GPU: ", tf.test.is_gpu_available())

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.

def run_selfplay(config: MuZeroConfig,
                 storage: SharedStorage,
                 replay_buffer: ReplayBuffer,
                 iteration: int):
    mcts = MCTS()
    for _ in tqdm(range(config.batch_size), desc="Selfplay Batch creation", position=1, leave=False):
        network = storage.latest_network()
        game = mcts.play_game(config,network)
        replay_buffer.save_game(game)
    print(f"Batch {iteration} creation completed. Starting training...")

# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.



def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return tf.losses.mean_squared_error(target, prediction)


def scale_gradient(tensor, scale: float):
    # Scales the gradient for the backward pass.
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch, weight_decay: float):
    with tf.GradientTape() as tape:

        loss = 0

        for (image, actions, targets) in batch:

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

    return network, loss

def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, iterations: int):
    network = storage.latest_network()
    learning_rate = config.lr_init * config.lr_decay_rate ** (iterations / config.lr_decay_steps)
    optimizer = tf.keras.optimizers.legacy.Adam()


    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.action_space_size)
    network, loss = update_weights(optimizer, network, batch, config.weight_decay)

    network.tot_training_steps += 1

    #save model every 25 episodes
    if iterations % 25 == 0 and iterations >0:
        network.backup_count += 1
        network.save_network_deepcopy(model_name=config.model_name)

    storage.save_network(network)
    tf.keras.backend.clear_session()

    return loss

def launch_job(f, *args):
    f(*args)


def muzero(config: MuZeroConfig):

    model_name = config.model_name
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)
    losses = []
    print(f"Starting Selfplay for {config.model_name}! Batch size is {config.batch_size}.")

    for i in tqdm(range(config.training_episodes), desc=f"Training episodes for {config.model_name}", position=0):

        # self-play
        launch_job(run_selfplay, config, storage, replay_buffer, i)

        # training
        loss = train_network(config, storage, replay_buffer, i)

        # print and plot loss
        print('Loss: ' + str(loss))
        losses.append(loss[0])
        plt.plot(losses, label=f"Loss {model_name}")
        plt.ylabel("Loss")
        plt.xlabel("batches processed")
        plt.show()
        plt.savefig("Plots/loss_plot_" + model_name + ".png")

if __name__ == "__main__":
    muzero(make_Go7x7_config())
    muzero(make_Go9x9_config())

