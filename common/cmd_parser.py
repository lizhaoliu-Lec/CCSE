import argparse


def parse_cmd_arg():
    parser = argparse.ArgumentParser(description='Parser to read your config file')
    parser.add_argument('--config', type=str, help='path to config file')
    return parser.parse_args()
