from asyncio.queues import Queue
import numpy as np
from top.env.reversi_env import ReversiEnv, Player, Winner
from collections import defaultdict, namedtuple
from top.util.bitboard import find_correct_moves, bit_to_array, flip_vertical, dirichlet_noise_of_mask, rotate90
from top.util.config import Config
from top.model.residual_network import ReversiModel


MCTSInfo = namedtuple('MCTSInfo', 'N W P')
StateKey = namedtuple('StateKey', 'black white curr_player')
ActionWithEvaluation = namedtuple("ActionWithEvaluation", "action n q")


class ReversiPlayer:
    def __init__(self, model: ReversiModel, config: Config, mcts_info=None, enable_resign=True, play_config=None):
        self.model = model
        self.enable_resign = enable_resign
        self.config = config
        self.play_config = play_config or self.config.play
        mcts_info = mcts_info if mcts_info else ReversiPlayer.create_mcts_infomation()
        self.N, self.W, self.P = mcts_info
        # N(s, a): visit times
        # W(s, a): value of action
        # P(s, a): possibility of policy
        # Q(s, a): average value of action

        self.expanded = set(self.P.keys())
        self.now_expanding = set()
        self.requested_stop_thinking = False
        self.resigned = False
        self.moves = []
        #self.loop = asyncio.get_event_loop()
        self.prediction_queue = Queue(self.play_config.prediction_queue_size)
        #self.sem = asyncio.Semaphore(self.play_config.parallel_search_num)

    def Q(self, key):
        return self.W[key] / (self.N[key] + 1e-5)

    def action(self, me, enemy):
        action_with_eval = self.action_with_evaluation(me, enemy)
        return action_with_eval.action

    def action_with_evaluation(self, me, enemy):
        env = ReversiEnv().set(me, enemy, Player.Black)  # the state of game currently.
        key = self.get_state_key(env)
        #print(key)
        # one think: do N simulation
        #env.render()
        for t in range(self.play_config.thinking_loop):
            if env.turn > 0:
                #print('inside??')
                self.search_moves(me, enemy)  # do N simulation
            else:
                #assert key.black == me
                legal_array = bit_to_array(find_correct_moves(me, enemy), 64)
                #print(legal_array)
                action = np.argmax(legal_array)
                self.N[key][action] = 1
                self.W[key][action] = 0
                self.P[key] = legal_array / np.sum(legal_array)
                #print(self.P[key])

            policy = self.calc_policy(me, enemy)
            action = int(np.random.choice(range(64), p=policy))
            action_by_value = int(np.argmax(self.Q(key) + (self.N[key] > 0) * 100))
            value_diff = self.Q(key)[action] - self.Q(key)[action_by_value]

            if env.turn <= self.play_config.start_rethinking_turn or self.requested_stop_thinking or \
                    (value_diff > -0.01 and self.N[key][action] >= self.play_config.required_visit_to_decide_action):
                break

        if self.play_config.resign_threshold is not None and \
                np.max(self.Q(key) - (self.N[key] == 0) * 10) <= self.play_config.resign_threshold:
            self.resigned = True
            if self.enable_resign:
                if env.turn >= self.config.play.allowed_resign_turn:
                    return ActionWithEvaluation(None, 0, 0)  # means resign

        saved_policy = self.calc_policy_by_tau_1(key) if self.config.play_data.save_policy_of_tau_1 else policy
        self.add_data_to_move_buffer_with_8_symmetries(me, enemy, saved_policy)
        return ActionWithEvaluation(action=action, n=self.N[key][action], q=self.Q(key)[action])

    def add_data_to_move_buffer_with_8_symmetries(self, me, enemy, policy):
        for flip in [False, True]:
            for rot_right in range(4):
                own_saved, enemy_saved, policy_saved = me, enemy, policy.reshape((8, 8))
                if flip:
                    own_saved = flip_vertical(own_saved)
                    enemy_saved = flip_vertical(enemy_saved)
                    policy_saved = np.flipud(policy_saved)
                if rot_right:
                    for _ in range(rot_right):
                        own_saved = rotate90(own_saved)
                        enemy_saved = rotate90(enemy_saved)
                    policy_saved = np.rot90(policy_saved, k=-rot_right)
                self.moves.append([(own_saved, enemy_saved), list(policy_saved.reshape((64, )))])

    def search_moves(self, me, enemy):
        # one simulation:
        #   1. select until a left node
        #   2. expand and evaluate
        #   3. back-propagate
        loop = self.loop
        self.running_simulation_num = 0
        self.requested_stop_thinking = False
        #coroutine_list = []

        for i in range(self.play_config.simulation_num_per_move):
            self.start_do_simulation(me, enemy, is_root_node=True)

        #coroutine_list.append(self.prediction_worker())
        #loop.run_until_complete(asyncio.gather(*coroutine_list))

    def start_do_simulation(self, me, enemy, is_root_node=True):
        self.running_simulation_num += 1
        key = self.get_state_key(ReversiEnv().set(me, enemy, Player.Black))
        if self.requested_stop_thinking:
            self.running_simulation_num -= 1
            return None
        env = ReversiEnv().set(me, enemy, Player.Black)
        value = self.do_simulation(env, is_root_node)
        self.running_simulation_num -= 1
        return value

    def do_simulation(self, env, is_root_node=True):
        key = self.get_state_key(env)
        if env.done:  # terminal node, return directly.
            if env.winner == Winner.Black:
                return 1
            elif env.winner == Winner.White:
                return -1
            else:
                return 0
        another_side_key = self.get_state_key(env, another_side=True)


        #while key in self.now_expanding:
        #    asyncio.sleep(self.config.play.wait_for_expanding_sleep_sec)

        if key not in self.expanded:  # reach a leaf node
            value = self.expand_and_evaluate(env)  # expand this node
            return value if env.curr_player == Player.Black else -value

        virtual_loss = self.play_config.virtual_loss
        virtual_loss_for_w = virtual_loss if env.next_player == Player.Black else -virtual_loss

        aciton = self.select(env, is_root_node)  # use PUCT algorithm.

        env.step(aciton)

        self.W[key][aciton] -= virtual_loss_for_w
        self.N[key][aciton] += virtual_loss  # set virtual loss

        value = self.do_simulation(env)

        self.N[key][aciton] -= virtual_loss - 1
        self.W[key][aciton] += virtual_loss_for_w + value

        self.N[another_side_key][aciton] += 1
        self.W[another_side_key][aciton] -= value
        return value  # 3. back-propagate

    def select(self, env, is_root_node):
        key = self.get_state_key(env)
        if env.curr_player == Player.Black:
            legal_moves = find_correct_moves(env.board.black, env.board.white)
        else:
            legal_moves = find_correct_moves(env.board.white, env.board.black)
        # PUCT = Q(s, a) + U(s, a)
        # U(s, a) = Cpuct * P(s, a) * sqrt(sum(N(s, b)) / (1 + N(s, a))
        sqrt = np.sqrt(np.sum(self.N[key]))
        sqrt = max(sqrt, 1)

        #print(self.P[key].shape, bit_to_array(legal_moves, 64).shape)
        posibility = self.P[key] * bit_to_array(legal_moves, 64)

        if np.sum(posibility) > 0:
            temperature = min(np.exp(1 - np.power(env.turn / self.play_config.policy_decay_turn
                                                  , self.play_config.policy_decay_power)), 1)
            # normalize and decay policy
            posibility = self.normalize(posibility, temperature)

        if is_root_node and self.play_config.noise_eps > 0:  # dirichlet noise
            noise = dirichlet_noise_of_mask(legal_moves, self.play_config.dirichlet_alpha)
            posibility = (1 - self.play_config.noise_eps) * posibility \
                         + self.play_config.noise_eps * noise

        U = self.play_config.c_puct * posibility * sqrt / (1 + self.N[key])

        if env.curr_player == Player.Black:
            PUCT = (self.Q(key) + U + 1000) * bit_to_array(legal_moves, 64)
        else:
            PUCT = (-self.Q(key) + U + 1000) * bit_to_array(legal_moves, 64)

        action = int(np.argmax(PUCT))
        return action

    def expand_and_evaluate(self, env: ReversiEnv):
        key = self.get_state_key(env)
        another_side_key = self.get_state_key(env, another_side=True)
        self.now_expanding.add(key)
        black, white = env.board.black, env.board.white

        is_flip_vertical = np.random.random() < 0.5
        rotate_right_num = int(np.random.random() * 4)
        if is_flip_vertical:
            black, white = flip_vertical(black), flip_vertical(white)
        for _ in range(rotate_right_num):
            black, white = rotate90(black), rotate90(white)  # rotate right

        black_image = bit_to_array(black, 64).reshape((8, 8))
        white_image = bit_to_array(white, 64).reshape((8, 8))
        state = [black_image, white_image]
        policy, value = self.model.predict(np.array(state))
        policy = np.array(policy).reshape((8, 8))
        if rotate_right_num > 0 or is_flip_vertical:
            if rotate_right_num > 0:
                policy = np.rot90(policy, k=rotate_right_num)
            if is_flip_vertical:
                policy = np.flipud(policy)
        policy = policy.reshape((64, ))
        self.P[key] = policy
        self.P[another_side_key] = policy
        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(value)

    def calc_policy(self, me, enemy):
        pc = self.play_config
        env = ReversiEnv().set(me, enemy, Player.Black)
        key = self.get_state_key(env)
        if env.turn < pc.change_tau_turn:
            return self.calc_policy_by_tau_1(key)
        else:
            action = np.argmax(self.N[key])  # tau = 0
            ret = np.zeros(64)
            ret[action] = 1
            return ret

    def finish_game(self, result):
        for move in self.moves:  # add this game winner result to all past moves.
            move += [result]

    def calc_policy_by_tau_1(self, key):
        return self.N[key] / np.sum(self.N[key])  # tau = 1

    @staticmethod
    def get_state_key(env, another_side=False):
        if another_side:
            return StateKey(black=env.board.white, white=env.board.black, curr_player=env.next_player.value)
        return StateKey(black=env.board.black, white=env.board.white, curr_player=env.curr_player.value)

    @staticmethod
    def normalize(p, t=1):
        pp = np.power(p, t)
        return pp / np.sum(pp)

    @staticmethod
    def create_mcts_infomation():
        return MCTSInfo(defaultdict(lambda: np.zeros(64)),
                        defaultdict(lambda: np.zeros(64)), defaultdict(lambda: np.zeros(64)))
