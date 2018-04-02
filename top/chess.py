from top.util.bitboard import bit_to_array, find_correct_moves
from top.util.config import Config
from top.trainer import get_model_dirs, load_best_model_weight
from top.env.reversi_player import ReversiPlayer, Player, Winner
from top.env.reversi_env import ReversiEnv
from top.model.residual_network import ReversiModel
import os
import numpy as np


def start(config: Config):
    chess = Chess(config)
    chess.start()


class Chess:
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def start(self, computer_color=Player.Black):
        if self._load_model():
            self._start_chess(computer_color)

    def _load_model(self):
        self.model = ReversiModel(self.config)
        if self.config.resource.use_best_model:
            return load_best_model_weight(self.model) is not None
        else:
            dirs = get_model_dirs(self.config.resource)
            if len(dirs) == 0:
                print('no model can be used.')
                return False
            latest_dir = dirs[-1]
            config_path = os.path.join(latest_dir, self.config.resource.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, self.config.resource.next_generation_model_weight_filename)
            self.model.load(config_path, weight_path)
            print('restore model from {}.'.format(latest_dir))
            return True

    @staticmethod
    def get_all_legal_moves(env):
        me, enmey = env.get_players()
        legal_moves = bit_to_array(find_correct_moves(me, enmey), 64)
        return legal_moves

    def _start_chess(self, computer_color=Player.Black):
        self.config.play.simulation_num_per_move = 300
        self.config.play.thinking_loop = 3
        computer = ReversiPlayer(self.model, self.config)
        env = ReversiEnv()
        env.reset()
        while not env.done:
            env.render()
            observation = env.observation
            if env.curr_player == computer_color:
                if computer_color == Player.Black:
                    action = computer.action(observation.black, observation.white)
                else:
                    action = computer.action(observation.white, observation.black)
            else:
                i = None
                illgeal_moves = self.get_all_legal_moves(env)
                while i is None or (int(illgeal_moves[i * 8 + j]) == 0):
                    i, j = input('input the coordinate(x, y): ').split()
                    i = int(i)
                    j = int(j)
                print(i, j)
                action = i * 8 + j
            env.step(action)