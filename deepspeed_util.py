import sys
import argparse


def get_argument_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--finetune',
                        default=False,
                        action='store_true',
                        help="Whether to finetune only")

    parser.add_argument(
        '--config',
        type=str,
        default='./config.json',
        help='Configuration path of json file  for pretraining GPT-3 '
    )

    return parser


def is_time_to_exit(args, epoch_steps=0, global_steps=0):
    return (epoch_steps >= args.max_steps_per_epoch) or \
            (global_steps >= args.max_steps)