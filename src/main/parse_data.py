from dataclasses import dataclass, field
import json
import os
import pickle
import sys
from typing import Sequence


@dataclass
class Caption:
    text: str
    train: bool


@dataclass
class ImageCaptions:
    captions: Sequence[Caption] = field(default_factory=list)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index=0) -> Caption:
        return self.captions[index]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    def load(path):
        with open(path, 'rb') as f:
            captions = pickle.load(f)
            f.close()
        return captions


def get_captions(train: Sequence, valid: Sequence) -> ImageCaptions:
    captions = []
    for caption in train['annotations']:
        caption = caption['caption']
        captions.append(Caption(text=caption, train=True))

    for caption in valid['annotations']:
        caption = caption['caption']
        captions.append(Caption(text=caption, train=False))
    return ImageCaptions(captions=captions)


def parse_captions(raw_data_path, parsed_data_path):
    train_path = os.path.join(raw_data_path, 'captions_train2014.json')
    valid_path = os.path.join(raw_data_path, 'captions_val2014.json')

    with open(train_path) as json_file:
        captions_train = json.load(json_file)

    with open(valid_path) as json_file:
        captions_val = json.load(json_file)

    captions = get_captions(captions_train, captions_val)
    captions.save(parsed_data_path)


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = json.load(open(config_path, 'r'))

    raw_data_path = config['task_file_paths']['raw_data_path']
    parsed_data_path = config['task_file_paths']['parsed_data_path']

    parse_captions(raw_data_path, parsed_data_path)