import gym_woodoku
import gymnasium as gym
from agents.single_agents import RandomAgent, ReflexAgent
from agents.multi_agents import MinmaxAgent, ExpectimaxAgent
import argparse
from game import Game
from comet_ml import Experiment
from comet_ml.integration.gymnasium import CometLogger
from dotenv import load_dotenv
import os

SUMMARY_ITERS = 5

agents = {
    "random": RandomAgent,
    "reflex": ReflexAgent,
    "minimax": MinmaxAgent,
    "expectimax": ExpectimaxAgent,
}

render_modes = {
    "GUI": "human",
    "RGB": "rgb_array",
    "Text": "ansi",
    "SummaryDisplay": None,
    "Testing": None,
}

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


def configure_logger(env):
    """
    Configure comet logger
    :return: comet logger
    """
    load_dotenv('./../.env')
    experiment = Experiment(
        api_key=os.getenv("API_TOKEN"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORKSPACE_NAME"),
    )
    return CometLogger(env, experiment)


def main():
    args = parse_args()

    env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode=render_modes[args.render])
    # env = configure_logger(env)

    game = Game(env, agents[args.agent]())

    iters = 1 if args.render != "SummaryDisplay" else SUMMARY_ITERS
    scores = []
    for i in range(iters):
        scores.append(game.run())
        print(f"Score: {scores[-1]}")

    if args.render == "SummaryDisplay":
        print(f"Average score over {SUMMARY_ITERS} iterations: {sum(scores) / SUMMARY_ITERS}")


if __name__ == "__main__":
    main()
    #input("Press Enter to continue...")