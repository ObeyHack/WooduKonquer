import os
import neptune
from dotenv import load_dotenv
from game import GameState, Agent, Game
from src.RLgame import RLAgent


class GameRunner(object):
    def __init__(self, env, agent, num_runs, agent_type, should_log=False, should_train=False):
        self.env = env
        self.agent = agent
        self.num_episodes = num_runs
        self.render_mode = self.env.env.env.render_mode
        self.logger = None
        if should_log:
            self.logger = self.configure_logger(agent_type)

        self.should_train = should_train
        self.game = Game(self.env, self.agent)

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
        self.env.env.env.render_mode = None

        # Train the agent
        print("Training the agent")
        RLAgent.train_agent(self.agent, self.env, num_episodes=1000, plot_rewards=True)

        # Turn on rendering
        self.env.env.env.render_mode = self.render_mode

    def play(self):
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

        if self.logger:
            self.logger.stop()

