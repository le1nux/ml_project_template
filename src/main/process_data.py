import json
import sys

from src.main.parse_data import ImageCaptions,Caption


def process_captions(parsed_data_path, processed_data_path):
    captions = ImageCaptions.load(parsed_data_path)

    for caption in captions:
        caption.text = caption.text.strip(" ")

    captions.save(processed_data_path)


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = json.load(open(config_path, 'r'))

    parsed_data_path = config['task_file_paths']['parsed_data_path']
    processed_data_path = config['task_file_paths']['processed_data_path']

    process_captions(parsed_data_path, processed_data_path)