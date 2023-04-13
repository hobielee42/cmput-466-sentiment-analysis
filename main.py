import os
import re
import shutil
import string

import tensorflow as tf
from keras import losses
from keras.layers import TextVectorization
from keras.utils import text_dataset_from_directory

batch_size = 32


def get_data():
    """
    Download and extract the IMDB dataset.
    """
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')
    os.listdir(train_dir)
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)


def custom_standardization(input_data):
    """
    Standardize the data by lowercasing and removing punctuation.
    :param input_data:
    :return:
    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# main
raw_train_ds = text_dataset_from_directory('aclImdb/train',
                                           batch_size=batch_size,
                                           validation_split=0.2,
                                           subset='training',
                                           seed=42)

raw_val_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train',
                                                        batch_size=batch_size,
                                                        validation_split=0.2,
                                                        subset='validation',
                                                        seed=42)

raw_test_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/test',
                                                         batch_size=batch_size)
vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        output_mode='tf_idf')

vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)])

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

model.summary()

epochs = 10
history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)
