from top.util.config import Config
from top.trainer import get_model_dirs, load_best_model_weight
from top.env.reversi_player import ReversiPlayer, Player, Winner
from top.env.reversi_env import ReversiEnv
from top.model.residual_network import ReversiModel
import os
import numpy as np


# evaluate which model is better
def start(config: Config):
    evaluator = Evaluator(config)
    evaluator.start()


class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.test_model = None
        self.best_model = None
        self.test_model_path = None

    def start(self):
        if self._load_model():
            self._start_evaluate()

    def _load_model(self):
        self.best_model = ReversiModel(self.config)
        self.test_model = ReversiModel(self.config)
        dirs = get_model_dirs(self.config.resource)
        if load_best_model_weight(self.best_model):
            print('restore best model: ', self.best_model.config.resource.model_best_weight_path)
        else:
            print('fail to restore best model: ', self.best_model.config.resource.model_best_weight_path)
            return False
        if len(dirs) == 0:
            print('fail to restore best model.')
            return False
        latest_dir = dirs[-1]
        self.test_model_path = latest_dir
        config_path = os.path.join(latest_dir, self.config.resource.next_generation_model_config_filename)
        weight_path = os.path.join(latest_dir, self.config.resource.next_generation_model_weight_filename)
        self.test_model.load(config_path, weight_path)
        print('restore model from {}'.format(weight_path))
        return True

    def _start_evaluate(self):
        best_player = ReversiPlayer(self.best_model, self.config, play_config=self.config.eval.play_config)
        test_player = ReversiPlayer(self.test_model, self.config, play_config=self.config.eval.play_config)
        winning_rate = 0
        env = ReversiEnv()
        for game_idx in range(self.config.eval.game_num):
            best_is_black = np.random.random() < 0.5
            if best_is_black:
                black_player, white_player = best_player, test_player
            else:
                black_player, white_player = test_player, best_player
            env.reset()
            observation = env.observation
            while not env.done:
                if env.curr_player == Player.Black:
                    action = black_player.action(observation.black, observation.white)
                else:
                    action = white_player.action(observation.white, observation.black)
                observation = env.step(action)
            black, white = env.board.number_of_stones
            if env.winner == Winner.Black:
                if best_is_black:
                    print(('Winner: Best Model(Black)', (black, white)))
                    winning_rate += 1
                else:
                    print(('Winner: Test Model(Black)', (black, white)))
            else:
                if best_is_black:
                    print(('Winner: Test Model(White)', (black, white)))
                else:
                    print(('Winner: Best Model(White)', (black, white)))
                    winning_rate += 1
        print('Best Model winning rate: {}'.format(1.0 * winning_rate / self.config.eval.game_num))
        print('Test Model winning rate: {}'.format(1 - 1.0 * winning_rate / self.config.eval.game_num))
        print('Test Model is {}'.format(self.test_model_path))
