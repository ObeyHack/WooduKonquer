import gym_woodoku
import gymnasium as gym
from agents import randomAgent, humanAgent
import argparse



agents = {
    "random": randomAgent,
    "human": humanAgent
}

render_modes = {
    "GUI": "human",
    "RGB": "rgb_array",
    "Text": "ansi",
    "SummaryDisplay": None,
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

    #
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode=render_modes[args.render])
    agents[args.agent].play(env)
    return


if __name__ == "__main__":
    main()
    #input("Press Enter to continue...")