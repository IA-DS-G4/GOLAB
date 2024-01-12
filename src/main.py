import tensorflow as tf
import keras
from muzeroconfig import MuZeroConfig
from mcts import NetworkStorage, GameStorage, MCTS
from matplotlib import pyplot as plt
from nn_models import Network
from Wrappers import Action, ActionHistory
from Go_7x7 import make_Go7x7_config
from Go_9x9 import make_Go9x9_config
from tqdm import tqdm
import pandas as pd

# to check and install GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("TensorFlow is using GPU: ", tf.test.is_gpu_available())


# We reapeteatly create moves from mcts
def run_selfplay(config: MuZeroConfig,
                 storage: NetworkStorage,
                 game_storage: GameStorage,
                 iteration: int):
    mcts = MCTS()
    for _ in tqdm(range(config.batch_size), desc="Selfplay Batch creation", position=1, leave=True):
        # load network from network save
        network = storage.latest_network()
        game = mcts.play_game(config,network)
        #store the game inside of a container
        game_storage.save_game(game)
    print(f"Batch {iteration} creation completed. Starting training...")





def scalar_loss(prediction, target) -> float:
    # calculating mean squared error
    return tf.losses.mean_squared_error(target, prediction)

def scale_gradient(tensor, scale: float):
    # Scales the gradient for the backward pass.
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)

# Function to apply gradients to update weights
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
            # compare prediciton and target
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
                    cce = keras.losses.CategoricalCrossentropy()
                    l_c = cce([target_policy], policy_t)

                l = l_a + l_b + l_c

                loss += scale_gradient(l, gradient_scale)

        loss /= len(batch)

        for weights in network.get_weights():
            loss += weight_decay * tf.nn.l2_loss(weights)

    # calculate gradients
    gradients = tape.gradient(loss, [network.representation.trainable_variables,
                                     network.dynamics.trainable_variables,
                                     network.policy.trainable_variables,
                                     network.value.trainable_variables,
                                     network.reward.trainable_variables])
    # apply grads on different networks
    optimizer.apply_gradients(zip(gradients[0], network.representation.trainable_variables))
    optimizer.apply_gradients(zip(gradients[1], network.dynamics.trainable_variables))
    optimizer.apply_gradients(zip(gradients[2], network.policy.trainable_variables))
    optimizer.apply_gradients(zip(gradients[3], network.value.trainable_variables))
    optimizer.apply_gradients(zip(gradients[4], network.reward.trainable_variables))

    return network, loss

def train_network(config: MuZeroConfig, storage: NetworkStorage, game_storage:GameStorage, iterations: int):
    network = storage.latest_network()
    # we decrease learning rate over time to stabilize training
    learning_rate = config.lr_init * config.lr_decay_rate ** (iterations / config.lr_decay_steps)
    optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    # load batch from game storage for training
    batch = game_storage.sample_batch(config.num_unroll_steps, config.td_steps, config.action_space_size)
    network, loss = update_weights(optimizer, network, batch, config.weight_decay)

    network.tot_training_steps += 1

    #save model every 10 episodes
    if iterations % 10 == 0 :
        network.backup_count += 1
        network.save_network_deepcopy(model_name=config.model_name)
    # save network after training
    storage.save_network(network)
    # to save some RAM
    keras.backend.clear_session()

    return loss


def muzero_train(config: MuZeroConfig):

    model_name = config.model_name
    storage = NetworkStorage(config)
    game_storage = GameStorage(config)
    losses = []
    print(f"Starting Selfplay for {config.model_name}! Batch size is {config.batch_size}.")

    for i in tqdm(range(config.training_episodes), desc=f"Training episodes for {config.model_name}", position=0):

        # self-play
        run_selfplay(config, storage, game_storage, i)

        # training
        loss = train_network(config, storage, game_storage, i)

        # print and plot loss
        print('Loss: ' + str(loss))
        losses.append(loss[0])
        plt.plot(losses, label=f"Loss {model_name}")
        plt.ylabel("Loss")
        plt.xlabel("batches processed")
        plt.show()
        plt.savefig("loss_plot_" + model_name + ".png")
        df = pd.DataFrame(losses, columns=['loss'])
        # saving the dataframe
        df.to_csv(f'loss_{model_name}.csv')


if __name__ == "__main__":
    # uncomment depending on which model you want to train
    muzero_train(make_Go7x7_config())
    #muzero(make_Go9x9_config())

