import numpy as np
import math
from hex_engine import HexPosition


def ucb_score(node, c=1.0):
    if node.visit_count == 0:
        return np.inf
    return node.total_value / node.visit_count + c * math.sqrt(math.log(node.parent.visit_count) / node.visit_count)


def select_child(node):
    ucb_scores = {
        action: ucb_score(child) for action, child in node.children.items()
    }
    best_action = max(ucb_scores, key=ucb_scores.get)
    return node.children[best_action]


def select(node):
    while node.expanded:
        node = select_child(node)
    return node


class GenericMCTS:
    def __init__(self, agent, game, root):
        self.agent = agent
        self.game = game
        self.root = root

    def run(self, size, num_simulations, action_list):
        for _ in range(num_simulations):
            node = select(self.root)
            value = self._expand(node, action_list)
            self._backpropagation(node, value)

        # Choose the action with the highest visit count from the root node
        best_action = max(self.root.children, key=lambda a: self.root.children[a].visit_count)
        return best_action

    def _expand(self, node, action_list):
        state = node.state
        legal_actions = action_list

        policy, value = self.agent.forward(state)
        policy = policy.cpu().detach().numpy()
        board_size = len(state)

        policy = np.resize(policy, (board_size, board_size))

        for action in legal_actions:
            action_index = HexPosition(board_size).coordinate_to_scalar(action)
            node.add_child(action, policy[action_index // board_size, action_index % board_size])

        return value[0].item()

    def _backpropagation(self, node, value):
        node.visit_count += 1
        node.total_value += value

        if node.parent is not None:
            self._backpropagation(node.parent, -value)


class A2CNode:
    def __init__(self, state, legal_actions, parent=None, prior=None):
        self.state = state
        self.legal_actions = legal_actions
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.expanded = False
        self.prior = prior

    def add_child(self, action, prior):
        self.children[action] = A2CNode(state=self.state, legal_actions=self.legal_actions, parent=self, prior=prior)
