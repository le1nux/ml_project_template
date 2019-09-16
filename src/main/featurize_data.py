import json
import sys

import h5py
from sklearn.feature_extraction.text import TfidfVectorizer

from src.main.parse_data import ImageCaptions,Caption


def get_tfidf_features(captions, featurized_data_path, max_features=10000):
    train_captions = [caption.text for caption in captions if caption.train]
    valid_captions = [caption.text for caption in captions if not caption.train]
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, max_features=max_features)

    tfidf_vectorizer.fit(train_captions)

    tfidf_train = tfidf_vectorizer.transform(train_captions[:10000]).toarray()
    tfidf_valid = tfidf_vectorizer.transform(valid_captions[:2000]).toarray()

    hf = h5py.File(featurized_data_path, 'w')
    grp = hf.create_group('captions')
    grp.create_dataset('train', data=tfidf_train)
    grp.create_dataset('valid', data=tfidf_valid)
    hf.close()


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = json.load(open(config_path, 'r'))

    processed_data_path = config['task_file_paths']['processed_data_path']
    featurized_data_path = config['task_file_paths']['featurized_data_path']
    max_features = config['featurize_data']['tfidf']['max_features']

    captions = ImageCaptions.load(processed_data_path)
    get_tfidf_features(captions, featurized_data_path, max_features=max_features)