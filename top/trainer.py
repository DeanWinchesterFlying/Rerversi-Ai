from datetime import datetime
from top.util.bitboard import bit_to_array
from top.util.config import Config, ResourceConfig
import os
from glob import glob
from top.model.residual_network import ReversiModel
from collections import Counter
from keras.optimizers import SGD
import json
import numpy as np
import tensorflow as tf
import keras.backend as K


# use the play data to train.
def set_session_config(per_process_gpu_memory_fraction=None, allow_growth=None):
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction,
            allow_growth=allow_growth,
        )
    )
    sess = tf.Session(config=config)
    K.set_session(sess)


def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=0.65)
    Trainer(config).start()


def get_model_dirs(rc: ResourceConfig):
    dir_pattern = os.path.join(rc.next_generation_model_dir,
                               rc.next_generation_model_dirname_tmpl % "*")
    dirs = list(sorted(glob(dir_pattern)))
    return dirs


def get_game_data_filenames(rc: ResourceConfig):
    pattern = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % "*")
    files = list(sorted(glob(pattern)))
    return files


def load_best_model_weight(model, clear_session=False):
    if clear_session:
        K.clear_session()
    return model.load(model.config.resource.model_best_config_path,
                      model.config.resource.model_best_weight_path)

def read_game_data_from_file(path):
    with open(path, "rt") as f:
        return json.load(f)

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: ReversiModel
        self.loaded_filenames = set()
        self.loaded_data = {}
        self.training_count_of_files = Counter()
        self.dataset = None
        self.optimizer = None

    def start(self):
        self.load_and_compile_model()
        self.start_train()

    def load_and_compile_model(self):
        model = ReversiModel(self.config)
        dirs = get_model_dirs(self.config.resource)
        if not dirs:
            if not load_best_model_weight(model) or not self.config.resource.use_best_model:
                model.bulid()
            else:
                print('load best model: ', model.config.resource.model_best_weight_path)
        else:
            latest_dir = dirs[-1]
            config_path = os.path.join(latest_dir, self.config.resource.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, self.config.resource.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
            print('restore model from {}'.format(weight_path))
        self.model = model
        self.optimizer = SGD(lr=1e-2, momentum=0.9)
        losses = [ReversiModel.policy_loss, ReversiModel.value_loss]
        self.model.model.compile(optimizer=self.optimizer, loss=losses)

    def start_train(self):
        total_steps = self.config.trainer.start_total_steps

        self.load_play_data()
        if self.dataset_size < self.config.trainer.min_data_size_to_learn:
            print('dataset_size is too small: {}'.format(self.dataset_size))
        self.update_learning_rate(total_steps)
        total_steps += self.train_epoch(self.config.trainer.epoch_to_checkpoint)
        print('train step: ', total_steps)
        self.save_model()
        print('save model successfully.')

    def remove_data(self):
        print('begin to remove train data.')
        for filename in self.loaded_filenames:
            self.training_count_of_files[filename] += 1
            if os.path.exists(filename):
                try:
                    print("remove {}".format(filename))
                    os.remove(filename)
                except Exception as e:
                    print(e)

    def load_play_data(self):
        filenames = get_game_data_filenames(self.config.resource)
        updated = False
        for filename in filenames:
            if filename not in self.loaded_filenames:
                try:
                    print("loading data from {}".format(filename))
                    data = read_game_data_from_file(filename)
                    self.loaded_data[filename] = self.convert_to_training_data(data)
                    self.loaded_filenames.add(filename)
                    updated = True
                except Exception as e:
                    print(str(e))

        for filename in (self.loaded_filenames - set(filenames)):
            try:
                print("loading data from {}".format(filename))
                data = read_game_data_from_file(filename)
                self.loaded_data[filename] = self.convert_to_training_data(data)
                self.loaded_filenames.add(filename)
            except Exception as e:
                print(str(e))
            updated = True

        if updated:
            state_ary_list, policy_ary_list, value_ary_list = [], [], []
            for s_ary, p_ary, v_ary in self.loaded_data.values():
                state_ary_list.append(s_ary)
                policy_ary_list.append(p_ary)
                value_ary_list.append(v_ary)

            state_ary = np.concatenate(state_ary_list)
            policy_ary = np.concatenate(policy_ary_list)
            value_ary = np.concatenate(value_ary_list)
            self.dataset = (state_ary, policy_ary, value_ary)

    @staticmethod
    def convert_to_training_data(data):
        state_list = []
        policy_list = []
        value_list = []
        for state, policy, value in data:
            me, enemy = bit_to_array(state[0], 64)\
                             .reshape((8, 8)), bit_to_array(state[1], 64).reshape((8, 8))
            state_list.append([me, enemy])
            policy_list.append(policy)
            value_list.append(value)

        return np.array(state_list), np.array(policy_list), np.array(value_list)

    def update_learning_rate(self, total_steps):
        for step, lr in self.config.trainer.lr_schedules:
            if total_steps >= step:
                ret = lr
        lr = ret
        if lr:
            K.set_value(self.optimizer.lr, lr)
            print("total step={}, set learning rate to {}".format(total_steps, lr))

    def train_epoch(self, epochs):
        state_ary, policy_ary, value_ary = self.dataset
        print('size: ', state_ary.shape, policy_ary.shape, value_ary.shape)
        self.model.model.fit(state_ary, [policy_ary, value_ary],
                             batch_size=self.config.trainer.batch_size,
                             epochs=epochs)
        steps = (state_ary.shape[0] // self.config.trainer.batch_size) * epochs
        return steps

    def save_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir,
                                 rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset[0])
