class MuZeroConfig(object):

    def __init__(self,
                 action_space_size: int,
                 observation_space_size: int,
                 observation_space_shape: (int,int),
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 dropout_rate: float,
                 training_episodes: int,
                 hidden_layer_size: int,
                 model_name: str,
                 visit_softmax_temperature,
                 ):
        ### Self-Play
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size
        self.observation_space_shape = observation_space_shape
        self.num_actors = num_actors

        self.visit_softmax_temperature = visit_softmax_temperature
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        # We discard old games after every bage to save on RAM memory
        self.window_size = batch_size
        self.batch_size = batch_size
        # num of steps top unroll for MCTS
        self.num_unroll_steps = 3
        self.td_steps = td_steps
        # optimization parameters
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.training_episodes = training_episodes
        # Network params
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = 0.1

        self.model_name = model_name

    def new_game(self):
        return Game(self.action_space_size, self.discount)

class Game:
    def __init__(self, action_space_size, discount):
        self.action_space_size = action_space_size
        self.discount = discount
        self.observation_history = []
        self.action_history = []
        self.rewards = []