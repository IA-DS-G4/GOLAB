import pygame
from pygame.locals import *
from go_board import GoBoard
from go_utils import GoUtils
from go_graphics import RenderGo
from MuZeroAgent import MuZeroAgent
from Go_9x9 import Go9x9Config,make_Go9x9_config
import numpy as np
# Constants


colors = {
    "b": (0, 0, 0),
    "w": (245, 245, 245),
    "r": (133, 42, 44),
    "y": (255, 204, 153),
    "g": (26, 81, 79)
}

WIDTH = 90
MARGIN = 2
PADDING = 50
DOT = 4



# Players
PLAYER_BLACK = 1
PLAYER_WHITE = -1

class GoGame:
    def __init__(self, config, dim):
        self.board = GoBoard(board_dimension=dim ,player=1)
        self.board_render = (WIDTH + MARGIN) * (dim - 1) + MARGIN  # Actual width for the board
        self.game_width = (WIDTH + MARGIN) * (dim - 1) + MARGIN + PADDING * 2
        self.game_height = self.game_width + 50
        pygame.init()
        pygame.font.init()
        self.display_surface = pygame.display.set_mode((self.game_width, self.game_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption(f'{config.model_name}')
        self.renderer = RenderGo(self)
        self.utils = GoUtils()
        self.pass_button_clicked = False
        self.passed_once = False
        self.game_over = False
        self.running = True
        self.playing = False
        self.win = False
        self.last_position = [-1, -1]
        self.agent = MuZeroAgent(config=config)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

        pos = pygame.mouse.get_pos()
        if self.playing and event.type == pygame.MOUSEBUTTONDOWN and self.mouse_in_pass_button(pos):
            self.pass_button_clicked = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if self.mouse_in_button(pos):
                if not self.playing:
                    self.start()
                    # MuZero plays first move
                    self.muzero_respond()
                else:
                    self.surrender()
                    self.board.flip_player()
            elif self.mouse_in_pass_button(pos) and self.playing:
                self.pass_button_clicked = False
                _, self.board = self.utils.make_move(board=self.board, move=(-1,-1))
                if not self.passed_once:
                    self.passed_once = True
                    self.renderer.render_all()

                    # MuZero plays
                    self.muzero_respond()
                    self.lastPosition = self.board.get_last_position()

                else:
                    # Double Pass Game Over
                    print("Game Over!")
                    self.game_over = True


            elif self.playing:
                c = (pos[0] - PADDING + WIDTH // 2) // (WIDTH + MARGIN)
                r = (pos[1] - PADDING + WIDTH // 2) // (WIDTH + MARGIN)

                if 0 <= r < self.board.board_dimension and 0 <= c < self.board.board_dimension:
                    is_valid, self.board = self.utils.make_move(board=self.board, move=(r, c))
                    if is_valid:
                        self.passed_once = False
                        self.lastPosition = self.board.get_last_position()
                        self.renderer.render_all()

                        # Machine plays
                        self.agent.receive_move((r,c))
                        self.muzero_respond()
                        self.lastPosition = self.board.get_last_position()
                    else:
                        print("Invalid move!")

    def on_execute(self):
        while self.running:
            self.board_init()
            for event in pygame.event.get():
                self.handle_event(event)
            self.renderer.render_all()
            pygame.display.update()
        pygame.quit()

    def start(self):
        self.playing = True
        self.last_position = [-1, -1]
        self.board = GoBoard(board_dimension=self.board.board_dimension, player=PLAYER_BLACK)
        self.win = False

    def surrender(self):
        self.playing = False
        self.win = True

    def board_init(self):
        self.display_surface.fill(colors["y"])
        # Draw black background rect for game area
        pygame.draw.rect(self.display_surface, colors["b"],
                         [PADDING,
                          PADDING,
                          self.board_render,
                          self.board_render])

        # Draw the grid
        for row in range(self.board.board_dimension - 1):
            for column in range(self.board.board_dimension - 1):
                pygame.draw.rect(self.display_surface, colors["y"],
                                 [(MARGIN + WIDTH) * column + MARGIN + PADDING,
                                  (MARGIN + WIDTH) * row + MARGIN + PADDING,
                                  WIDTH,
                                  WIDTH])
    def mouse_in_button(self, pos):
        return self.game_width // 2 - 50 <= pos[0] <= self.game_width // 2 + 50 and self.game_height - 85 <= pos[1] <= self.game_height - 55

    def mouse_in_pass_button(self, pos):
        return self.game_height // 2 - 50 <= pos[0] <= self.game_height // 2 + 50 and self.game_height - 50 <= pos[1] <= self.game_height - 20

    def handle_pass_button_click(self):
        self.pass_button_clicked = False
        _, self.board = self.utils.make_move(board=self.board, move=(-1,-1))
        if not self.passed_once:
            self.passed_once = True
        else:
            print("Game Over!")
            self.game_over = True
        self.print_winner()

    def handle_board_click(self, pos):
        col = (pos[0] - PADDING + WIDTH // 2) // (WIDTH + MARGIN)
        row = (pos[1] - PADDING + WIDTH // 2) // (WIDTH + MARGIN)

        if 0 <= row < self.board.board_dimension and 0 <= col < self.board.board_dimension:
            _, self.board = self.utils.make_move(board=self.board, move=(row, col))
            self.passed_once = False
            self.print_winner()
            self.last_position = self.board.get_last_position()

    def print_winner(self):
        winner, winning_by_points = self.utils.evaluate_winner(self.board.board_grid)
        print(f"Player {winner} wins by {winning_by_points} points!")

    def retrieve_winner(self):
        return self.utils.evaluate_winner(self.board.board_grid)

    def muzero_respond(self):
        print("machine responds")
        move,sting_move = self.agent.generate_move()
        valid, self.board = self.utils.make_move(board=self.board, move=move)


if __name__ == "__main__":
    go_game = GoGame(config=make_Go9x9_config(), dim=9)
    go_game.on_execute()
