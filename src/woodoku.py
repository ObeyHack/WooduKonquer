import argparse
from agents.single_agents import RandomAgent, ReflexAgent
from agents.multi_agents import MinmaxAgent, AlphaBetaAgent, ExpectimaxAgent
from agents.RL_agents.qlearning import QLearningAgent, ApproximateQAgent
from agents.RL_agents.mantoCarloAgent import MantoCarloAgent
from gameRunner import GameRunner

SUMMARY_ITERS = 10

agents = {
    "random": RandomAgent,
    "reflex": ReflexAgent,
    "minimax": MinmaxAgent,
    "alpha_beta": AlphaBetaAgent,
    "expectimax": ExpectimaxAgent,
    "q_learning": QLearningAgent,
    "q_approx": ApproximateQAgent,
    "manto_carlo": MantoCarloAgent,
}

render_modes = {
    "GUI": "human",
    "Video": "rgb_array",
    "Text": "ansi",
    "SummaryDisplay": None,
    "Testing": None,
}

RL_agents = ["q_learning", "q_approx", "manto_carlo"]


def parse_args():
    """
    Parse command line arguments
    agent: agent to play the game (random, human)
    render: render mode (human, rgb_array, ansi)
    :return: dictionary of arguments
    """
    parser = argparse.ArgumentParser(description='Play Woodoku Game')
    # agent
    parser.add_argument("-a", "--agent", dest="agent", type=str, choices=agents.keys(),
                        help="Agent to play the game", default="random")

    # render
    parser.add_argument("-d", "--display", dest="render", type=str, choices=render_modes.keys(),
                        help="Render mode", default="GUI")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    agent = agents[args.agent]()
    should_train = args.agent in RL_agents
    iters = 1 if args.render != "SummaryDisplay" else SUMMARY_ITERS
    should_log = (args.render == "SummaryDisplay")

    GameRunner(agent, iters, args.agent, render_modes[args.render], should_log=should_log,
                should_train=should_train).play()


if __name__ == "__main__":
    main()
    #input("Press Enter to continue...")