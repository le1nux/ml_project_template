import json
import os

config = json.load(open('config.json', 'r'))

for task,sub_config in config.items():
    if os.path.isfile(f'make_utils/{task}.json'):
        task_config = json.load(open(f'make_utils/{task}.json', 'r'))
    else:
        task_config = None

    if sub_config != task_config and task != 'task_file_paths':
        json.dump(sub_config, open(f'make_utils/{task}.json', 'w'))