import math
import random
from time import time as clock
from copy import deepcopy
from src.search import Node
from src.game import Agent, GameState


class MCSTNode(Node):
    """
      Node for the MCTS. Stores the move applied to reach this node from its parent,
      stats for the associated game position, children, parent and outcome
      (outcome==none unless the position ends the game).
      Args:
          move:
          parent:
          N (int): times this position was visited
          Q (int): average reward (wins-losses) from this position
          children (dict): dictionary of successive nodes
          outcome (int): If node is a leaf, then outcome indicates
                         the winner, else None
      """

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """
        Initialize a new node with optional move and parent and initially empty
        children list and rollout statistics and unspecified outcome.
        """
        super().__init__(state, parent=parent, action=action, path_cost=path_cost)

        self.N = 0  # times this position was visited
        self.Q = 0  # average reward (wins-losses) from this position
        self.children = {} # dictionary of successive nodes - action:node

    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf node
        """
        return len(self.children) == 0

    @property
    def times_visited(self):
        return self.N

    @property
    def total_reward(self):
        return self.Q

    @property
    def value(self, explore: float = 0.5):
        """
        Calculate the UCT value of this node relative to its parent, the parameter
        "explore" specifies how much the value should favor nodes that have
        yet to be thoroughly explored versus nodes that seem to have a high win
        rate.
        """

        # if the node is not visited, set the value as infinity. Nodes with no visits are on priority
        if self.N == 0:
            return 0 if explore == 0 else float('inf')
        else:
            return self.Q / self.N + explore * math.sqrt(2 * math.log(self.parent.N) / self.N)  # exploitation + exploration

    def generate_child_node(self, action):
        """
        Return the child node with the given action.
        """
        child_state = self.state.generate_successor(action)
        return MCSTNode(child_state, parent=self, action=action)

    def populate_children(self, expend_num=50) -> None:
        """Creates and populates the children of this node by randomly
        selecting moves and determining the resultant state
        """
        legal_actions = self.state.get_legal_actions()
        random.shuffle(legal_actions)

        for action in legal_actions[:expend_num]:
            child_node = self.generate_child_node(action)
            self.children[action] = child_node


class MCST_agent(Agent):
    def __init__(self, time_limit=3):
        super().__init__()
        self.time_limit = time_limit

    def get_action(self, state):
        # Implement the getAction method here
        start_time = clock()
        root = MCSTNode(state)

        while clock() - start_time < self.time_limit:
            leaf = self.select_node(root)
            leaf.populate_children()
            reward = self.rollout(leaf)
            self.backpropagate(leaf, reward)

        action = self.get_best_child_action(root)
        return action

    def select_node(self, root_node: MCSTNode) -> MCSTNode:
        """
        Select a node in the tree to preform a single simulation from.
        """
        node = root_node
        # stop if we find reach a leaf node
        while not node.is_leaf():
            # descend to the maximum value node, break ties at random
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value).value
            max_nodes = [n for n in node.children.values()
                         if n.value == max_value]

            node = random.choice(max_nodes)
        return node

    def rollout(self, current_node: Node) -> int:
        """
        Conduct a playout from the current node.
        """
        state = current_node.state
        return len(list(state.get_legal_actions()))

    def backpropagate(self, node: MCSTNode, reward: int) -> None:
        """
        Update the node statistics back up to the root.
        """
        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent

    def get_best_child_action(self, node: MCSTNode) -> int:
        """Determines the action that is expected to return the highest valuation
        from the given node

        Args:
            node (MCSTNode): The node from which the actions are being executed

        Returns:
            action (int): The action that is expected to return the highest valuation
        """
        best_child = list(node.children.items())[0][1]
        best_action = list(node.children.items())[0][0]

        for action, child in node.children.items():
            if child.times_visited == 0:
                continue

            if best_child.total_reward / best_child.times_visited < child.total_reward / child.times_visited:
                best_child = child
                best_action = action

        return best_action






