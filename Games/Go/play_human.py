import pygame
from pygame.locals import *
from go_board import GoBoard
from go_utils import GoUtils
from go_graphics import render_go


BOARD_DIM = 5  # Define an x by x board

# Define colors
BLACK = (0, 0, 0)
WHITE = (245, 245, 245)
RED = (133, 42, 44)
YELLOW = (208, 176, 144)
GREEN = (26, 81, 79)

PLAYER_BLACK = 1
PLAYER_WHITE = -1
EMPTY = 0
PASS = (-1, -1)

# Define grid globals
WIDTH = 20  # Width of each square on the board
MARGIN = 1  # How thick the lines are
PADDING = 50  # Distance between board and border of the window
DOT = 4  # Number of dots
BOARD = (WIDTH + MARGIN) * (BOARD_DIM - 1) + MARGIN  # Actual width for the board
GAME_WIDTH = BOARD + PADDING * 2
GAME_HIGHT = GAME_WIDTH + 100


class Go:
    def __init__(self):
        self.go_board = GoBoard(board_dimension=BOARD_DIM, player=PLAYER_BLACK)
        pygame.init()
        pygame.font.init()
        self._display_surf = pygame.display.set_mode((GAME_WIDTH, GAME_HIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Go')
        self.render = render_go(self)
        self.utils = GoUtils()
        self._running = True
        self._playing = False
        self._win = False
        self.lastPosition = [-1, -1]
        self.pass_button_clicked = False
        self.passed_once = False
        self.game_over = False

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

        pos = pygame.mouse.get_pos()
        if self._playing and event.type == pygame.MOUSEBUTTONDOWN and self.mouse_in_pass_button(pos):
            self.pass_button_clicked = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if self.mouse_in_botton(pos):
                if not self._playing:
                    self.start()
                else:
                    self.surrender()
                    self.go_board.flip_player()
            elif self._playing and self.mouse_in_pass_button(pos):
                self.pass_button_clicked = False
                _, self.go_board = self.utils.make_move(board=self.go_board, move=PASS)
                if not self.passed_once:
                    self.passed_once = True
                else:
                    # Double Pass Game Over
                    print("Game Over!")
                    self.game_over = True

                self.print_winner()

            elif self._playing:
                c = (pos[0] - PADDING + WIDTH // 2) // (WIDTH + MARGIN)
                r = (pos[1] - PADDING + WIDTH // 2) // (WIDTH + MARGIN)

                if 0 <= r < BOARD_DIM and 0 <= c < BOARD_DIM:
                    _, self.go_board = self.utils.make_move(board=self.go_board, move=(r, c))
                    self.passed_once = False
                    self.print_winner()
                    self.lastPosition = self.go_board.get_last_position()

        # print(self.go_board)
        # print()


    def on_execute(self):
        while (self._running):
            self.go_board_init()
            for event in pygame.event.get():
                self.on_event(event)
            self.render.render_all()
            pygame.display.update()
        pygame.quit()

    def start(self):
        self._playing = True
        self.lastPosition = [-1, -1]
        self.go_board = GoBoard(board_dimension=BOARD_DIM, player=PLAYER_BLACK)
        self._win = False

    def surrender(self):
        self._playing = False
        self._win = True

    def go_board_init(self):
        self._display_surf.fill(YELLOW)
        # Draw black background rect for game area
        pygame.draw.rect(self._display_surf, BLACK,
                         [PADDING,
                          PADDING,
                          BOARD,
                          BOARD])

        # Draw the grid
        for row in range(BOARD_DIM - 1):
            for column in range(BOARD_DIM - 1):
                pygame.draw.rect(self._display_surf, YELLOW,
                                 [(MARGIN + WIDTH) * column + MARGIN + PADDING,
                                  (MARGIN + WIDTH) * row + MARGIN + PADDING,
                                  WIDTH,
                                  WIDTH])
    def mouse_in_botton(self, pos):
        """ Check if mouse is in the button and return a boolean value
        """
        if GAME_WIDTH // 4 * 3 - 50 <= pos[0] <= GAME_WIDTH // 4 * 3 + 50 and GAME_HIGHT - 50 <= pos[
            1] <= GAME_HIGHT - 20:
            return True
        return False

    def mouse_in_pass_button(self, pos):
        """ Check if mouse is in the pass button and return a boolean value
        """
        if GAME_WIDTH // 4 - 50 <= pos[0] <= GAME_WIDTH // 4 + 50 and GAME_HIGHT - 50 <= pos[1] <= GAME_HIGHT - 20:
            return True
        return False


    def print_winner(self):
        winner, winning_by_points = self.utils.evaluate_winner(self.go_board.board_grid)
        if winner == PLAYER_BLACK:
            print("Black wins by " + str(winning_by_points))
        else:
            print("White wins by " + str(winning_by_points))

    def retrieve_winner(self):
        return self.utils.evaluate_winner(self.go_board.board_grid)


if __name__ == "__main__":
    go = Go()
    go.on_execute()