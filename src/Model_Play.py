import pygame
from pygame.locals import *
from go_board import GoBoard
from go_utils import GoUtils
from go_graphics import RenderGo
import tensorflow as tf
from nn_models import Network
from muzeroconfig import MuZeroConfig
import socket
import random
import time
from mcts import MCTS
from Wrappers import Action, Player, Node
import numpy as np

# Constants
BOARD_SIZE = 9
WIDTH = 90
MARGIN = 2
PADDING = 50
DOT = 4
BOARD = (WIDTH + MARGIN) * (BOARD_SIZE - 1) + MARGIN # Actual width for the board
GAME_WIDTH = (WIDTH + MARGIN) * (BOARD_SIZE - 1) + MARGIN + PADDING * 2
GAME_HEIGHT = GAME_WIDTH + 50

colors = {
    "b": (0, 0, 0),
    "w": (245, 245, 245),
    "r": (133, 42, 44),
    "y": (255, 204, 153),
    "g": (26, 81, 79)
}

# Players
PLAYER_BLACK = 1
PLAYER_WHITE = -1
PASS = (-1, -1)

class MuZeroAgent:
    def __init__(self, model_path, game, config: MuZeroConfig):
        self.model_path = model_path
        self.config = config
        self.game = game

        # Assuming you have an instance of Network
        self.network = Network(config=self.config)

        # Load the weights for a specific model and backup
        model_name_to_load = 'model1'
        backup_count_to_load = 1  # Adjust this based on the specific backup count you want to load

        self.network.load_network_deepcopy(model_name=model_name_to_load, backup_count=backup_count_to_load)

        # Now, the `network` instance has its weights loaded from the specified backup.

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
        action = MCTS.select_action(self.config, len(self.game.action_history), root, self.network)
        self.game.store_search_statistics(root)
        # print(f"move{action} from Player {game.board.player}")
        self.game.apply(action)
        r = int(np.floor(action.index / self.game.board_size))
        c = int(action.index % self.game.board_size)
        move = (r, c)
        if action == self.game.board_size ** 2:
            move = (-1, -1)
        return f"MOVE {move[0]},{move[1]}"

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
                move = self.generate_random_move()
                time.sleep(1)
                client_socket.send(move.encode())
                print("Send:", move)

                # Wait for server response
                response = client_socket.recv(1024).decode()
                print(f"Server Response1: {response}")
                if "END" in response: break

            first = False
            response = client_socket.recv(1024).decode()
            print(f"Server Response2: {response}")
            if "END" in response: break

            # Add some condition to break the loop, if necessary
            # Example: If server sends a certain message, or after a number of moves

        client_socket.close()

class PlayClientServer:
    def __init__(self):
        self.board = GoBoard(board_dimension=BOARD_SIZE, player=PLAYER_BLACK)
        pygame.init()
        pygame.font.init()
        self.display_surface = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Go')
        self.renderer = RenderGo(self)
        self.utils = GoUtils()
        self.pass_button_clicked = False
        self.passed_once = False
        self.game_over = False
        self.running = True
        self.playing = False
        self.win = False
        self.last_position = [-1, -1]
        self.agent = MuZeroAgent(model_path="../models", restored=True)



if __name__ == "__main__":
    go_game = GoGame()
    go_game.on_execute()
