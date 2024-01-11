import pygame

BOARD_SIZE = 9

# Colors
colors = {
    "b": (0, 0, 0),
    "w": (245, 245, 245),
    "r": (133, 42, 44),
    "y": (208, 176, 144)
}

PLAYER_BLACK = 1
PLAYER_WHITE = -1
PASS = (-1, -1)

# Grid Globals
WIDTH = 90
MARGIN = 2
PADDING = 50
DOT = 4
BOARD = (WIDTH + MARGIN) * (BOARD_SIZE - 1) + MARGIN # Actual width for the board
BOARD_WIDTH = (WIDTH + MARGIN) * (BOARD_SIZE - 1) + MARGIN
GAME_WIDTH = BOARD_WIDTH + PADDING * 2
GAME_HEIGHT = GAME_WIDTH + 50

class RenderGo:
    def __init__(self, go_game):
        self.go_game = go_game

    def _render_button(self):
        color = colors["b"] if not self.go_game.playing else colors["r"]
        info = "Start" if not self.go_game.playing else "Surrender"

        pygame.draw.rect(self.go_game.display_surface, color, (GAME_WIDTH // 2 - 50, GAME_HEIGHT - 85, 100, 30))

        info_font = pygame.font.SysFont('Helvetica', 16)
        text = info_font.render(info, True, colors["w"])
        text_rect = text.get_rect()
        text_rect.centerx = GAME_WIDTH // 2
        text_rect.centery = GAME_HEIGHT - 75
        self.go_game.display_surface.blit(text, text_rect)

    def _render_pass_button(self):
        color = colors["b"] if not self.go_game.pass_button_clicked else colors["y"]
        info = "Pass"
        pygame.draw.rect(self.go_game.display_surface, color, (GAME_WIDTH // 2 - 50, GAME_HEIGHT - 50, 100, 30))
        info_font = pygame.font.SysFont('Helvetica', 16)
        text = info_font.render(info, True, colors["w"])
        text_rect = text.get_rect()
        text_rect.centerx = GAME_WIDTH // 2
        text_rect.centery = GAME_HEIGHT - 35
        self.go_game.display_surface.blit(text, text_rect)

    def _render_game_info(self):
        # Current player color
        if not self.go_game.game_over:
            color = colors["b"] if self.go_game.board.player == PLAYER_BLACK else colors["w"]
        else:
            color, win_by_points = self.go_game.retrieve_winner()

        center = (GAME_WIDTH // 8 - 50 , BOARD_WIDTH + 80)
        radius = 12

        pygame.draw.circle(self.go_game.display_surface, color, center, radius, 0)

        if not self.go_game.game_over:
            info = "Your Turn"
        else:
            info = f"wins by {win_by_points} points."

        info_font = pygame.font.SysFont('Helvetica', 16)
        text = info_font.render(info, True, colors["b"])
        text_rect = text.get_rect()
        text_rect.centerx = center[0] + 50
        text_rect.centery = center[1]
        self.go_game.display_surface.blit(text, text_rect)

    def _render_go_piece(self):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                center = ((MARGIN + WIDTH) * col + MARGIN + PADDING, (MARGIN + WIDTH) * row + MARGIN + PADDING)
                if self.go_game.board.board_grid[row][col] != 0:
                    color = colors["b"] if self.go_game.board.board_grid[row][col] == PLAYER_BLACK else colors["w"]
                    pygame.draw.circle(self.go_game.display_surface, color,
                                       center,
                                       WIDTH // 2 - MARGIN,
                                       0)

    def _render_last_position(self):
        if self.go_game.last_position[0] > 0 and self.go_game.last_position[1] > 0:
            pygame.draw.rect(self.go_game.display_surface, colors["r"],
                             ((MARGIN + WIDTH) * self.go_game.last_position[1] - (MARGIN + WIDTH) // 2 + PADDING,
                              (MARGIN + WIDTH) * self.go_game.last_position[0] - (MARGIN + WIDTH) // 2 + PADDING,
                              (MARGIN + WIDTH),
                              (MARGIN + WIDTH)), 1)

    def print_winner(self):
        winner, winning_by_points = self.go_game.utils.evaluate_winner(self.go_game.board.board_grid)
        player = "Black" if winner == PLAYER_BLACK else "White"
        print(f"{player} wins by {winning_by_points}")

    def render_all(self):
        self._render_go_piece()
        self._render_last_position()
        self._render_game_info()
        self._render_button()
        self._render_pass_button()
