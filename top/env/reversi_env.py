from enum import Enum
from ..util.bitboard import bit_count, calc_flip, find_correct_moves, board_to_string, bit_to_array

Player = Enum('Player', 'White Black')
Winner = Enum('Winner', 'Black White Draw')


class Board:
    def __init__(self, black=None, white=None):
        self.black = black or (0b00010000 << 24 | 0b00001000 << 32)
        self.white = white or (0b00001000 << 24 | 0b00010000 << 32)

    @property
    def number_of_stones(self):
        return bit_count(self.black), bit_count(self.white)


class ReversiEnv:
    def __init__(self):
        self.board = None
        self.curr_player = None
        self.turn = 0
        self.done = False
        self.winner = None

    def reset(self):
        self.board = Board()
        self.curr_player = Player.Black
        self.turn = 0
        self.done = False
        self.winner = None

    def set(self, black, white, next_player):
        self.board = Board(black, white)
        self.curr_player = Player.Black
        self.turn = sum(self.board.number_of_stones) - 4
        self.done = False
        self.winner = None
        return self

    def step(self, action):
        assert action is None or 0 <= action <= 63

        if action is None:  # give up the game
            self._let_another_player_win()
            self._game_over()
            return self.board

        me, enmey = self.get_players()
        '''moves = bit_to_array(find_correct_moves(me, enmey), 64)
        print(moves)
        for i, move in enumerate(moves):
            if move == 1:
                action = i
                break
        print('action is ', action)'''

        flipped = calc_flip(action, me, enmey)

        if bit_count(flipped) == 0: # illegal action
            print('illegal action')
            self._let_another_player_win()
            self._game_over()
            return self.board

        me ^= flipped  # flip enemy's stones
        me |= 1 << action  # take action
        enmey ^= flipped  # flip enemy's stones

        # update the board
        if self.curr_player == Player.Black:
            self.board.black, self.board.white = me, enmey
        else:
            self.board.white, self.board.black = me, enmey

        self.turn += 1
        if bit_count(find_correct_moves(int(enmey), int(me))):
            # next player has at least one legal action to take.
            self.switch_to_next_player()
        elif bit_count(find_correct_moves(int(me), int(enmey))) == 0:
            # current player has no legal action to take.
            self._game_over()  # both players have no legal action to take.
        return self.board

    def get_players(self):
        if self.curr_player == Player.Black:
            me, enemy = self.board.black, self.board.white
        else:
            enemy, me = self.board.black, self.board.white
        return int(me), int(enemy)

    def switch_to_next_player(self):
        self.curr_player = self.next_player

    @property
    def next_player(self):
        return Player.White if self.curr_player == Player.Black else Player.Black

    @property
    def observation(self):
        return self.board

    @property
    def current_player(self):
        return self.curr_player

    def render(self):
        b, w = self.board.number_of_stones
        print("next={} turn={} B={} W={}".format(self.next_player.name, self.turn, b, w))
        print(board_to_string(self.board.black, self.board.white, with_edge=True))

    def _let_another_player_win(self):
        if self.next_player == Player.Black:
            self.winner = Winner.Black
        else:
            self.winner = Winner.White

    def _game_over(self):
        self.done = True
        if self.winner is None:
            black_num, white_num = self.board.number_of_stones
            if black_num > white_num:
                self.winner = Winner.Black
            elif black_num < white_num:
                self.winner = Winner.White
            else:
                self.winner = Winner.Draw