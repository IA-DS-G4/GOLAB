import numpy as np
#from self_play.mcts import MCTS

class SelfPlay():
    """Algorithm plays against itself till the game ends and produce a set of (board, policy, result)
    Used as training data for the neural net.
    """
    def __init__(self, starting_board, nn, utils, simluation_number):
        self.utils = utils
        self.nn = nn
        self.simluation_number = simluation_number
        self.current_board = starting_board
        self.policies = np.empty(0)
        self.history_boards = np.empty(0) #Records all the board config played in this self play session

    def play_one_move(self):
        ts_instance = MCTS(self.current_board, self.nn, self.utils, self.simluation_number)
        new_board, move, policy = ts_instance.run_all_simulations(temp1 = 1, temp2 = 0.0, step_boundary=5)

        print("move is:", move)
        if len(self.policies) == 0:
            self.policies = policy
        else:
            self.policies = np.vstack((self.policies, policy))

        self.history_boards = np.append(self.history_boards, self.current_board) #Save current board to history
        self.current_board = new_board #Update current board to board after move

        return move == (-1, -1)

    def play_till_finish(self):
        move_num = 0
        #Cut the game if we played for too long
        while (not self.utils.is_game_finished(self.current_board)) and move_num <= self.current_board.board_dimension**2 * 2:
            self.play_one_move()
            move_num += 1

        boards_data = np.array([augment_board for history_board in self.history_boards for augment_board in history_board.generate_augmented_boards()])
        reversed_boards_data = [b.reverse_board_config() for b in boards_data]

        winner, _ = self.utils.evaluate_winner(self.current_board.board_grid)
        #corresponding winner for each history board from current perspective
        new_training_labels_v = np.array([[winner] if history_board.player != self.current_board.player else [-winner] \
            for history_board in self.history_boards])
        new_training_labels_v = np.repeat(new_training_labels_v, 5, axis=0)
        new_training_labels_v = np.append(new_training_labels_v, new_training_labels_v, axis=0)
        new_training_labels_p = np.repeat(np.array(self.policies), 5, axis=0)
        new_training_labels_p = np.append(new_training_labels_p, new_training_labels_p, axis=0)

        return np.append(boards_data, reversed_boards_data), new_training_labels_p, new_training_labels_v
        #return boards_data, new_training_labels_p, new_training_labels_v
