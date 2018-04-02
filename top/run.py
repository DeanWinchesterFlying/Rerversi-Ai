import argparse

from top import self_play, trainer, evaluator, chess
from moke_config import create_config
from top.util.config import Config


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true',
                        help='prepare data')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model')
    parser.add_argument('--play', action='store_true',
                        help='evaluate the model')
    parser.add_argument('--best', action='store_true',
                        help='use the best the model')
    return parser

parser = create_parser()
args = parser.parse_args()
config = create_config(Config)

config.resource.use_best_model = args.best
if args.prepare:
    self_play.start(config)
if args.train:
    trainer.start(config)
if args.evaluate:
    evaluator.start(config)
if args.play:
    chess.start(config)

'''config = create_config(Config)
self_play.start(config)'''