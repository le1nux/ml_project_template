import json
import pickle
import sys

config_path = sys.argv[1]
config = json.load(open(config_path, 'r'))

parsed_data_path = config['task_file_paths']['parsed_data_path']
processed_data_path = config['task_file_paths']['processed_data_path']

parsed_data = pickle.load(open(parsed_data_path, 'rb'))
# Convert parsed_data to processed_data
processed_data = 'Processed Data'

pickle.dump(processed_data, open(processed_data_path, 'wb'))