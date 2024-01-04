from typing import Dict, List, Optional
class Action(object):

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

    def __str__(self):
        return str(self.index)


class Player(object):
    def __init__(self, index: int):
        self.index = index
    def __hash__(self):
        return self.index
    def __eq__(self, other):
        return self.index == other.index
    def __gt__(self, other):
        return self.index > other.index
    def __str__(self):
        return str(self.index)


class ActionHistory(object):
    """Simple history container used inside the search.
       Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int, player):
        self.history = history
        self.action_space_size = action_space_size
        self.player = player

    def clone(self):
        return ActionHistory(self.history, self.action_space_size, self.player)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]
    def to_play(self):
        return Player(self.player)


class Node(object):

    def __init__(self, prior: float):

        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:

        return len(self.children) > 0

    def value(self) -> float:

        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count