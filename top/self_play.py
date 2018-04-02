import json
from datetime import datetime
from time import time
from glob import glob
from top.env.reversi_env import ReversiEnv
from top.env.reversi_env import Player, Winner
from top.util.config import Config, ResourceConfig
import numpy as np
from top.env.reversi_player import ReversiPlayer
from top.model.residual_network import ReversiModel
import os
from top.trainer import get_model_dirs, load_best_model_weight


# self-play to gain train data.
def start(config: Config):
    env = ReversiEnv()
    model = ReversiModel(config)
    dirs = get_model_dirs(config.resource)
    if not dirs:
        if not load_best_model_weight(model) or not config.resource.use_best_model:
            model.bulid()
        else:
            print('load best model: ', model.config.resource.model_best_weight_path)
    else:
        latest_dir = dirs[-1]
        config_path = os.path.join(latest_dir, config.resource.next_generation_model_config_filename)
        weight_path = os.path.join(latest_dir, config.resource.next_generation_model_weight_filename)
        model.load(config_path, weight_path)
        print('restore model from {}'.format(weight_path))
    #model.load(config.resource.model_best_config_path)
    play_worker = SelfPlayWorker(config=config, env=env, model=model)
    play_worker.start()



def write_game_data_to_file(path, data):
    with open(path, "wt") as f:
        json.dump(data, f)


def read_game_data_from_file(path):
    with open(path, "rt") as f:
        return json.load(f)


def get_game_data_filenames(rc: ResourceConfig):
    pattern = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % "*")
    files = list(sorted(glob(pattern)))
    return files


class SelfPlayWorker:
    def __init__(self, env, model, config: Config):
        self.env = env
        self.config = config
        self.model = model
        self.black = None
        self.white = None
        self.buffer = []
        self.resign_test_game_count = 0
        self.false_positive_count_of_resign = 0

    def start(self):
        self._start()

    def _start(self):
        np.random.seed(None)
        self.buffer = []
        mtcs_info = None
        local_idx = 0

        while local_idx < 2000:
            np.random.seed(None)
            local_idx += 1

            if mtcs_info is None and self.config.play.share_mtcs_info_in_self_play:
                mtcs_info = ReversiPlayer.create_mcts_infomation()

            start_time = time()
            env = self.start_game(mtcs_info)
            time_spent = time() - start_time
            print("play game {} time={} sec, ".format(local_idx, time_spent))
            print("turn={}:{}:{}".format(env.turn, env.board.number_of_stones, env.winner))

            if self.config.play.reset_mtcs_info_per_game and local_idx % self.config.play.reset_mtcs_info_per_game == 0:
                mtcs_info = None

            with open(self.config.resource.self_play_game_idx_file, "wt") as f:
                f.write(str(local_idx))

    def start_game(self, mtcs_info):
        self.env.reset()
        enable_resign = self.config.play.disable_resignation_rate <= np.random.random()
        # self.config.play.simulation_num_per_move = ?
        self.black = self.create_reversi_player(enable_resign=enable_resign, mtcs_info=mtcs_info)
        self.white = self.create_reversi_player(enable_resign=enable_resign, mtcs_info=mtcs_info)

        # play game
        observation = self.env.observation
        while not self.env.done:
            if self.env.curr_player == Player.Black:
                action = self.black.action_with_evaluation(observation.black, observation.white)
            else:
                action = self.white.action_with_evaluation(observation.white, observation.black)
            observation = self.env.step(action.action)
            #print('step {}.'.format(action))

        self.finish_game(resign_enabled=enable_resign)
        self.save_play_data()
        self.remove_play_data()

        return self.env

    def save_play_data(self, write=True):
        # drop draw game by drop_draw_game_rate
        if self.black.moves[0][-1] != 0 or self.config.play_data.drop_draw_game_rate <= np.random.random():
            data = self.black.moves + self.white.moves
            self.buffer += data

        if not write or not self.buffer:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        write_game_data_to_file(path, self.buffer)
        self.buffer = []

    def remove_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        try:
            for i in range(len(files) - self.config.play_data.max_file_num):
                os.remove(files[i])
        except:
            pass

    def finish_game(self, resign_enabled=True):
        if self.env.winner == Winner.Black:
            black_win = 1
            false_positive_of_resign = self.black.resigned
        elif self.env.winner == Winner.White:
            black_win = -1
            false_positive_of_resign = self.white.resigned
        else:
            black_win = 0
            false_positive_of_resign = self.black.resigned or self.white.resigned

        self.black.finish_game(black_win)
        self.white.finish_game(-black_win)

        if not resign_enabled:
            self.resign_test_game_count += 1
            if false_positive_of_resign:
                self.false_positive_count_of_resign += 1
            self.check_and_update_resignation_threshold()

    def check_and_update_resignation_threshold(self):
        if self.resign_test_game_count < 100 or self.config.play.resign_threshold is None:
            return

        old_threshold = self.config.play.resign_threshold
        if self.false_positive_rate >= self.config.play.false_positive_threshold:
            self.config.play.resign_threshold -= self.config.play.resign_threshold_delta
        else:
            self.config.play.resign_threshold += self.config.play.resign_threshold_delta
        self.reset_false_positive_count()

    def reset_false_positive_count(self):
        self.false_positive_count_of_resign = 0
        self.resign_test_game_count = 0

    @property
    def false_positive_rate(self):
        if self.resign_test_game_count == 0:
            return 0
        return self.false_positive_count_of_resign / self.resign_test_game_count

    def create_reversi_player(self, enable_resign=None, mtcs_info=None):
        return ReversiPlayer(self.model, self.config, mtcs_info, enable_resign)

