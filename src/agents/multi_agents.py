import abc

from tqdm import tqdm

from src import util
from src.game import Agent
import numpy as np


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=1):
        self.evaluation_function = score_evaluation_function
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""

        legal_moves = game_state.get_legal_actions()
        # we are starting with the max player and we want to maximize the score of the game ,after that we will
        # minimize the score of the game for the opponent player and so on.
        # this is the first move of the max player
        best_move = max(tqdm(legal_moves), key=lambda x: self.minimax(game_state.generate_successor(x), 0, 1))
        return best_move

    def minimax(self, state, depth, agent_index):
        if depth == self.depth or state.is_terminated():
            return self.evaluation_function(state)

        if agent_index == 0:  # Our agent
            return max(self.minimax(state.generate_successor(action), depth + 1, 1)
                       for action in state.get_legal_actions())
        else:  # Opponent
            return min(self.minimax(state.generate_successor(action), depth + 1, 0)
                       for action in state.get_legal_actions())


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """ YOUR CODE HERE """

        def max_value(game_state, depth, alpha, beta):
            if depth == 0 or game_state.is_terminated():
                return self.evaluation_function(game_state)
            v = float('-inf')
            for action in game_state.get_legal_actions():
                v = max(v, min_value(game_state.generate_successor(action), depth, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(game_state, depth, alpha, beta):
            if depth == 0 or game_state.is_terminated():
                return self.evaluation_function(game_state)
            v = float('inf')
            for action in game_state.get_legal_actions():
                v = min(v, max_value(game_state.generate_successor(action), depth - 1, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        best_action = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in tqdm(game_state.get_legal_actions()):
            score = min_value(game_state.generate_successor(action), self.depth, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        legal_moves = game_state.get_legal_actions()
        best_move = max(legal_moves, key=lambda x: self.expectimax(game_state.generate_successor(x), 0, 1))
        return best_move


    def expectimax(self, state, depth, agent_index):
        if depth == self.depth or state.is_terminated():
            return self.evaluation_function(state)

        if agent_index == 0:  # Our agent
            return max(self.expectimax(state.generate_successor(action), depth + 1, 1)
                       for action in tqdm(state.get_legal_actions()))
        else:  # Opponent
            legal_actions = state.get_legal_actions()
            return sum(self.expectimax(state.generate_successor(action), depth + 1, 0) for action in
                       tqdm(legal_actions)) / len(legal_actions)
