from src.evaluations.evaluationStateFunctions import *


def best_evaluation_multi(current_game_state):
    """
    This evaluation function gets a game state and returns a combination of the functions connected_components
    and num_empty_squares

    :param current_game_state: the current game state
    :return: the number of 3x3 empty crushable squares on the board
    """
    return (15 * remaining_possible_moves(current_game_state) +
            connected_components(current_game_state) + square_contribution(current_game_state)**2)


def best_evaluation(current_game_state, action):
    """
    This evaluation function is an expression made up of various evaluation functions.
    The function gives the best average score for a ReflexAgent

    :param current_game_state: the current game state
    :param action: the action to evaluate
    :return: the score of the action

    .. seealso:: :class:`ReflexAgent`
    For more details, see the ReflexAgent class.
    """
    successor = current_game_state.generate_successor(action=action)
    return (connected_components(successor) + 15 * remaining_possible_moves(successor) +
            square_contribution(successor) ** 2)

