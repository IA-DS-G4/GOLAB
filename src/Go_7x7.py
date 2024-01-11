import numpy
from go_board import GoBoard
from go_utils import GoUtils
from muzeroconfig import MuZeroConfig, ActionHistory
from typing import List
from Wrappers import Action, Player, Node
import tensorflow as tf

class Go7x7Config(MuZeroConfig):

    def new_game(self):
        return Go7x7()

def make_Go7x7_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):

        # higher temperature higher exploration

        if training_steps < 100:
            return 3
        elif training_steps < 125:
            return 2
        elif training_steps < 150:
            return 1
        elif training_steps < 175:
            return 0.5
        elif training_steps < 200:
            return 0.250
        elif training_steps < 225:
            return 0.125
        else:
            return 0.125

    return Go7x7Config(action_space_size= 50,
                        observation_space_size= 49,
                        observation_space_shape= (7,7),
                        max_moves=98,
                        discount=0.999,
                        dropout_rate = 0.1,
                        dirichlet_alpha=0.25,
                        num_simulations=3,
                        batch_size=16,
                        td_steps=25,
                        lr_init=0.00001,
                        lr_decay_steps=10,
                        training_episodes=500,
                        hidden_layer_size= 49,
                        visit_softmax_temperature_fn=visit_softmax_temperature,
                        num_actors=2,
                        model_name="Go7x7")


class Go7x7:
    def __init__(self):
        self.board_size = 7
        self.player = 1 # Black goes first
        self.board = GoBoard(board_dimension=self.board_size, player=self.player)
        self.utils = GoUtils()
        self.observation_space_shape = (self.board_size,self.board_size)
        self.observation_space_size = self.board_size**2
        self.action_space_size = (self.board_size**2)+1
        self.action_history = []
        self.rewards = []
        self.observation_list = []
        self.child_visits = []
        self.root_values = []
        self.discount = 0.999
        self.done = False
    def clean_memory(self):
        del self.board

    def step(self, action):
        r = int(numpy.floor(action / self.board_size))
        c = int(action % self.board_size)
        move = (r,c)
        if action == self.board_size**2:
            move = (-1,-1)
        move_viable, self.board = self.utils.make_move(board=self.board,move=move)
        if not move_viable:
            done = True
            reward = -1
            return self.get_observation(), reward, done
        done = self.utils.is_game_finished(board=self.board)
        if done and move_viable:
            reward = 1 if self.utils.evaluate_winner(board_grid=self.board.board_grid)[0] == self.player else -1
        elif not done and move_viable:
            reward = 0
        return self.get_observation(), reward, done

    def apply(self, action: Action):
        observation, reward, done = self.step(action.index)
        self.rewards.append(reward)
        self.action_history.append(action.index)
        self.observation_list.append(observation)
        self.done = done


    def get_observation(self):
        return tf.constant([self.board.board_grid],dtype="int32")

    def legal_actions(self)-> List[Action]:
        # Pass = boardsize**2 is always legal
        legal = [self.board_size**2]
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.utils.is_valid_move(board=self.board,move=(i,j)):
                    legal.append(i * self.board_size + j)
        return [Action(index) for index in legal]

    def total_rewards(self):
        return sum(self.rewards)

    def is_finished(self):
        finished = self.utils.is_game_finished(board=self.board)
        if finished:
            print("game is finished!")
        return finished

    def to_play(self):
        return Player(self.board.player)

    def get_action_history(self) -> ActionHistory:

        return ActionHistory(self.action_history, self.action_space_size,self.board.player)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(action) for action in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player,
                    action_space_size: int):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i  # pytype: disable=unsupported-operands

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, last_reward, []))
        return targets


if __name__ == "__main__":
    pass