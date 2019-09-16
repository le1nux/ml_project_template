import datetime
import json
import os
import pickle
import sys

# from sacred import Experiment
# from sacred.observers import FileStorageObserver
#
# ex = Experiment()

config_path = sys.argv[1]

# ex.add_config(config_path)

config = json.load(open(config_path, 'r'))

featurized_data_path = config['task_file_paths']['featurized_data_path']

exp_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_dir = os.path.join('experiments', exp_name)
os.makedirs(model_dir, exist_ok=True)

featurized_data = pickle.load(open(featurized_data_path, 'rb'))
# Train model based on featurized_data
model = 'trained_model'
pickle.dump(model, open(f'{model_dir}/model.p', 'wb'))