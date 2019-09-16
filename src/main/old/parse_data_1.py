import json
import pickle
import sys

config_path = sys.argv[1]
config = json.load(open(config_path, 'r'))

raw_data_path = config['task_file_paths']['raw_data_path']
parsed_data_path = config['task_file_paths']['parsed_data_path']

raw_data = json.load(open(raw_data_path, 'r'))
# Convert raw_data to parsed_data
parsed_data = 'Parsed Data'

pickle.dump(parsed_data, open(parsed_data_path, 'wb'))