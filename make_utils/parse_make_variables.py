from argparse import ArgumentParser
import json

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-k',
        '--keys',
        help='config keys to get path of task data separated by /',
        required=False
    )

    return parser.parse_args()

config = json.load(open('config.json', 'r'))

args = parse_args()
keys = args.keys.split('/')

print(config[keys[0]][keys[1]])