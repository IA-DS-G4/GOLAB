from nn_models import Network
from muzeroconfig import MuZeroConfig
import socket
import random
import time
from mcts import MCTS
from Wrappers import Action, Player, Node
import numpy as np
from Go_7x7 import Go7x7, make_Go7x7_config
from Go_9x9 import Go9x9, make_Go9x9_config

class MuZeroAgent:
    def __init__(self, config: MuZeroConfig, backup_count_to_load=1):
        self.model_path = config.model_name
        self.config = config
        self.game = self.config.new_game()

        # Assuming you have an instance of Network
        self.network = Network(config=self.config)

        # Load the weights for a specific model and backup
        self.backup_count_to_load = backup_count_to_load

        self.network.load_network_deepcopy(model_name=self.model_path, backup_count=self.backup_count_to_load)

        # Now, the `network` instance has its weights loaded from the specified backup.

    def connect_to_server(self,host='localhost', port=12345):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))

        response = client_socket.recv(1024).decode()
        print(f"Server ResponseINIT: {response}")

        Game = response[-4:]
        print("Playing:", Game)

        if "1" in response:
            ag = 1
        else:
            ag = 2
        first = True

        while True:
            # Generate and send a random move

            if ag == 1 or not first:
                move, move_str = self.generate_move()
                time.sleep(1)
                client_socket.send(move_str.encode())
                print("Send:", move_str)

                # Wait for server response
                response = client_socket.recv(1024).decode()
                move = self.decode_move_string(response)
                print(f"Server Response1: {response}")
                self.act_other_agent_move(move)
                if "END" in response or self.game.is_finished(): break
            if ag == 2:
                response = client_socket.recv(1024).decode()
                move = self.decode_move_string(response)
                print(f"Server Response1: {response}")
                if "END" in response or self.game.is_finished(): break
                self.act_other_agent_move(move)
                move, move_str = self.generate_move()
                time.sleep(1)
                client_socket.send(move_str.encode())
                print("Send:", move_str)

        client_socket.close()

    def act_other_agent_move(self,move):
        action = Action(move[0]*self.game.board_size + move[1])
        self.game.apply(action)

    def generate_random_move(self):
        x = random.randint(0, 9)
        y = random.randint(0, 9)
        return f"MOVE {x},{y}"

    def generate_move(self):
        root = Node(0)
        current_observation = self.game.get_observation()
        MCTS.expand_node(root,
                         self.game.to_play(),
                         self.game.legal_actions(),
                         self.network.initial_inference(current_observation))
        MCTS.run_mcts(self.config, root, self.game, self.network)
        action = MCTS.select_action(self.config, root, self.network)
        self.game.store_search_statistics(root)
        # print(f"move{action} from Player {game.board.player}")
        self.game.apply(action)
        r = int(np.floor(action.index / self.game.board_size))
        c = int(action.index % self.game.board_size)
        move = (r, c)
        if action.index == self.game.board_size ** 2:
            move = (-1, -1)
        return move, f"MOVE {move[0]},{move[1]}"
    @staticmethod
    def decode_move_string(move_string):
        # Assuming move_string is in the format "MOVE x,y"
        _, coordinates = move_string.split(" ")
        x, y = map(int, coordinates.split(","))
        return x, y


if __name__ == "__main__":
    agent = MuZeroAgent(config=make_Go9x9_config())
    agent.connect_to_server()

