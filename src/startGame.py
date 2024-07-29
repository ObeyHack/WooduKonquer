import gym_woodoku
import gymnasium as gym
import agents.randomAgent


def main():
    # rgb_array for video, human for us
    env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode='human')

    # env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode='rgb_array')
    # env = gym.wrappers.RecordVideo(env, video_folder='./video_folder')

    #env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode='ansi')

    agents.randomAgent.play(env)


if __name__ == "__main__":
    main()
    #input("Press Enter to continue...")