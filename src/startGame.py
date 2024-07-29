import gym_woodoku
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode='human')
    # env = gym.wrappers.RecordVideo(env, video_folder='./video_folder')

    observation, info = env.reset()
    for i in range(100000):
        # here we need to place the action choice
        action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action)
        if terminated:
            env.reset()
    env.close()