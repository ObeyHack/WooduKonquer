import gymnasium as gym
import gym_woodoku
import os
import neptune
from dotenv import load_dotenv
from neptune.types import File

from game import Game
from src.RLgame import RLAgent


class GameRunner(object):
    def __init__(self, agent, num_runs, agent_type, render_mode, should_log=False, should_train=False):
        self._woodoku_env = None
        self.game = None
        self.env = None
        self.agent = agent
        self.num_episodes = num_runs
        self.render_mode = render_mode
        self.logger = None
        self.agent_type = agent_type
        if should_log:
            self.logger = self.configure_logger(self.agent_type)
        self.should_train = should_train

    def configure_logger(self, agent_type):
        """
        Configure comet logger
        :return: comet logger
        """
        load_dotenv('./../.env')
        API_TOKEN = os.environ.get("API_TOKEN")
        PROJECT_NAME = os.environ.get("PROJECT_NAME")

        logger_config = {
            "api_token": API_TOKEN,
            "project": PROJECT_NAME,
            "tags": [agent_type], # Add your tags here
        }
        run = neptune.init_run(**logger_config)
        return run

    def _log(self, score):
        self.logger["score"].append(score)

    def _train(self):
        self._woodoku_env.render_mode = None

        # Train the agent
        print("Training the agent")
        rewards = RLAgent.train_agent(self.agent, self.env, num_episodes=1000, plot_rewards=False)

        # Turn on rendering
        self._woodoku_env.render_mode = self.render_mode

        if self.logger:
            self.logger["rewards"].extend(rewards)

    def _log_videos(self):
        # all video files are stored in the video_folder named: {agent_type}*
        video_folder = "./video_folder"
        for file in os.listdir(video_folder):
            if file.startswith(self.agent_type):
                path = os.path.join(video_folder, file)
                self.logger[file].upload(path)

    def setup_env(self):
        self.env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode=self.render_mode)
        if self.render_mode == "rgb_array":
            self.env = gym.wrappers.RecordVideo(self.env, video_folder='./video_folder',
                                                episode_trigger=lambda x: x % 5 == 0, name_prefix=self.agent_type)

        self._woodoku_env = self.env.env.env
        if self.env.render_mode == "rgb_array":
            self._woodoku_env = self.env.env.env.env

        self.game = Game(self.env, self.agent)

    def play(self):
        self.setup_env()

        if self.should_train:
            self._train()

        scores = []
        for i in range(self.num_episodes):
            score = self.game.run()

            scores.append(score)
            if self.logger:
                self._log(scores[-1])

            print(f"Score: {scores[-1]}")

        print(f"Average score over {self.num_episodes} iterations: {sum(scores) / self.num_episodes}")

        self.env.close()
        
        if self.logger:
            self.logger["average_score"] = sum(scores) / self.num_episodes
            self._log_videos()
            self.logger.stop()

