import pygame

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

class render_go:
    def __init__(self,go_game):
        self.go_game = go_game


    def _render_button(self):
        color = GREEN if not self.go_game._playing else RED
        info = "Start" if not self.go_game._playing else "Surrender"

        pygame.draw.rect(self.go_game._display_surf, color,
                         (GAME_WIDTH // 4*3 - 50, GAME_HIGHT - 50, 100, 30))

        info_font = pygame.font.SysFont('Helvetica', 16)
        text = info_font.render(info, True, WHITE)
        textRect = text.get_rect()
        textRect.centerx = GAME_WIDTH // 4*3
        textRect.centery = GAME_HIGHT - 35
        self.go_game._display_surf.blit(text, textRect)

    def _render_pass_button(self):
        color = GREEN if not self.go_game.pass_button_clicked else YELLOW
        info = "Pass"

        pygame.draw.rect(self.go_game._display_surf, color,
                         (GAME_WIDTH // 4 - 50, GAME_HIGHT - 50, 100, 30))

        info_font = pygame.font.SysFont('Helvetica', 16)
        text = info_font.render(info, True, WHITE)
        textRect = text.get_rect()
        textRect.centerx = GAME_WIDTH // 4
        textRect.centery = GAME_HIGHT - 35
        self.go_game._display_surf.blit(text, textRect)

    def _render_game_info(self):
        #current player color
        if not self.go_game.game_over:
            color = BLACK if self.go_game.go_board.player == PLAYER_BLACK else WHITE
        else:
            color, win_by_points = self.go_game.retrieve_winner()

        center = (GAME_WIDTH // 2 - 60, BOARD + 60)
        radius = 12

        pygame.draw.circle(self.go_game._display_surf, color, center, radius, 0)

        if not self.go_game.game_over:
            info = "Wins!" if self.go_game._win else "Your Turn"
        else:
            info = "wins by " + str(win_by_points) + " points."
        info_font = pygame.font.SysFont('Helvetica', 16)
        text = info_font.render(info, True, BLACK)
        textRect = text.get_rect()
        textRect.centerx = self.go_game._display_surf.get_rect().centerx + 20
        textRect.centery = center[1]
        self.go_game._display_surf.blit(text, textRect)

    def _render_go_piece(self):
        """ Render the Go stones on the board according to self.go_board
        """
        # print('rendering go pieces')
        # print(self.go_board)
        for r in range(BOARD_DIM):
            for c in range(BOARD_DIM):
                center = ((MARGIN + WIDTH) * c + MARGIN + PADDING,
                          (MARGIN + WIDTH) * r + MARGIN + PADDING)
                if self.go_game.go_board.board_grid[r][c] != EMPTY:
                    color = BLACK if self.go_game.go_board.board_grid[r][c] == PLAYER_BLACK else WHITE
                    pygame.draw.circle(self.go_game._display_surf, color,
                                       center,
                                       WIDTH // 2 - MARGIN,
                                       0)


    def _render_last_position(self):
        """ Render a red rectangle around the last position
        """
        if self.go_game.lastPosition[0] > 0 and self.go_game.lastPosition[1] > 0:
            pygame.draw.rect(self.go_game._display_surf,RED,
                             ((MARGIN + WIDTH) * self.go_game.lastPosition[1] - (MARGIN + WIDTH) // 2 + PADDING,
                              (MARGIN + WIDTH) * self.go_game.lastPosition[0] - (MARGIN + WIDTH) // 2 + PADDING,
                              (MARGIN + WIDTH),
                              (MARGIN + WIDTH)),1)

    def print_winner(self):
        winner, winning_by_points = self.go_game.utils.evaluate_winner(self.go_game.go_board.board_grid)
        if winner == PLAYER_BLACK:
            print ("Black wins by " + str(winning_by_points))
        else:
            print ("White wins by " + str(winning_by_points))
    def render_all(self):

        self._render_go_piece()
        self._render_last_position()
        self._render_game_info()
        self._render_button()
        self._render_pass_button()

