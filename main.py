import os
import shutil

import tensorflow as tf
from keras.utils import text_dataset_from_directory


def get_data():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')
    os.listdir(train_dir)
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)


raw_train_ds = text_dataset_from_directory('aclImdb/train', batch_size=32, validation_split=0.2, subset='training',
                                           seed=42)
