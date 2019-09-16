import json
import pickle
import sys

config_path = sys.argv[1]
config = json.load(open(config_path, 'r'))

processed_data_path = config['task_file_paths']['processed_data_path']
featurized_data_path = config['task_file_paths']['featurized_data_path']

processed_data = pickle.load(open(processed_data_path, 'rb'))
# Convert processed_data to featurized_data
featurized_data = 'Featurized Data'

pickle.dump(featurized_data, open(featurized_data_path, 'wb'))