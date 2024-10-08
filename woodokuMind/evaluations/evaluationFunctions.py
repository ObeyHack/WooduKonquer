from woodokuMind.evaluations.evaluationStateFunctions import *


def best_evaluation_multi(current_game_state):
    """
    This evaluation function gets a game state and returns a combination of the functions connected_components
    and num_empty_squares

    :param current_game_state: the current game state
    :return: the number of 3x3 empty crushable squares on the board
    """
    score = (
             15 * int(remaining_possible_moves(current_game_state)) +
             1 * int(connected_components(current_game_state)) +
             1 * int(square_contribution(current_game_state)**2) -
             3 * int(avoid_jagged_edges(current_game_state)) +
             0 * int(score_function(current_game_state)) +
             0 * int(row_contribution(current_game_state)) +
             0 * int(column_contribution(current_game_state)) +
             2 * int(num_empty_squares(current_game_state)) -
             1 * int(avoid_jagged_edges_diag(current_game_state))
             )
    return score


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
    return best_evaluation_multi(successor)

