import gym_woodoku
import gymnasium as gym
from agents import randomAgent
import argparse


def parse_args():
    """
    Parse command line arguments
    agent: agent to play the game (random, human)
    render: render mode (human, rgb_array, ansi)
    :return: dictionary of arguments
    """
    parser = argparse.ArgumentParser(description='Play Woodoku Game')
    # agent
    parser.add_argument("-a", "--agent", dest="agent", type=str, choices=['random', 'human'],
                        help="Agent to play the game", default="random")

    # render
    parser.add_argument("-r", "--render", dest="render", type=str, choices=['human', 'rgb_array', 'ansi'],
                        help="Render mode", default="human")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    agents = {
        "random": randomAgent
    }

    env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode=args.render)
    agents[args.agent].play(env)
    return


if __name__ == "__main__":
    main()
    #input("Press Enter to continue...")