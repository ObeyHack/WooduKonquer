import abc
import numpy as np
from tqdm import tqdm
from woodokuMind import util
from woodokuMind.game import Agent
from woodokuMind.evaluations.evaluationFunctions import best_evaluation_multi


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
    def __init__(self, evaluation_function='best_evaluation_multi', depth=1):
        super().__init__()
        self.evaluation_function = util.lookup(evaluation_function, globals())
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

        legal_moves = game_state.get_legal_actions(agent_index=0)
        # we are starting with the max player and we want to maximize the score of the game ,after that we will
        # minimize the score of the game for the opponent player and so on.
        # this is the first move of the max player
        best_move = max(tqdm(legal_moves), key=lambda x: self.minimax(game_state.generate_successor(x, agent_index=0), 0, 1))
        return best_move

    def minimax(self, state, depth, agent_index):
        if depth == self.depth or state.is_terminated():
            return self.evaluation_function(state)

        if agent_index == 0:  # Our agent
            max_value = float('-inf')
            for action in state.get_legal_actions(agent_index=agent_index):
                value = self.minimax(state.generate_successor(action, agent_index=agent_index), depth + 1, 1)
                max_value = max(max_value, value)
            return max_value

        else:  # Opponent
            min_value = float('inf')
            for action in state.get_legal_actions(agent_index=agent_index):
                value = self.minimax(state.generate_successor(action, agent_index=agent_index), depth + 1, 0)
                min_value = min(min_value, value)
            return min_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def __init__(self, depth=2, **kwargs):
        super().__init__(depth=depth, **kwargs)

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """ YOUR CODE HERE """

        def max_value(game_state, depth, alpha, beta):
            if depth == 0 or game_state.is_terminated():
                return self.evaluation_function(game_state)
            v = float('-inf')
            for action in game_state.get_legal_actions(agent_index=0):
                v = max(v, min_value(game_state.generate_successor(action, agent_index=0), depth - 1, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(game_state, depth, alpha, beta):
            if depth == 0 or game_state.is_terminated():
                return self.evaluation_function(game_state)
            v = float('inf')
            for action in game_state.get_legal_actions(agent_index=1):
                v = min(v, max_value(game_state.generate_successor(action, agent_index=1), depth - 1, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        best_action = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in tqdm(game_state.get_legal_actions(agent_index=0)):
            score = min_value(game_state.generate_successor(action, agent_index=0), self.depth, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    def __init__(self, depth=1, **kwargs):
        super().__init__(depth=depth, **kwargs)
    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        legal_moves = game_state.get_legal_actions(agent_index=0)
        best_move = None
        best_val = float('-inf')
        for action in tqdm(legal_moves):
            next_state = game_state.generate_successor(action, agent_index=0)
            val = self.expectimax(next_state, 0, 1)
            if val > best_val:
                best_val = val
                best_move = action
        return best_move


    def expectimax(self, state, depth, agent_index):
        if depth == self.depth or state.is_terminated():
            return self.evaluation_function(state)

        if agent_index == 0:  # Our agent
            v = float('-inf')
            for action in state.get_legal_actions(agent_index=agent_index):
                next_state = state.generate_successor(action, agent_index=agent_index)
                curr_v = self.expectimax(next_state, depth + 1, 1)
                if curr_v > v:
                    v = curr_v
            return v

        else:  # Opponent
            legal_actions = state.get_legal_actions(agent_index=agent_index)
            v_sum = 0
            for action in legal_actions:
                next_state = state.generate_successor(action, agent_index=agent_index)
                v_sum += self.expectimax(next_state, depth + 1, 0)
            return v_sum / len(legal_actions)

