import os
import re
import shutil
import string

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import TextVectorization
from keras.utils import text_dataset_from_directory

batch_size = 32


def get_data():
    """
    Download and extract the IMDB dataset.
    """
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file(
            "aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
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


def nn_builder():
    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(vocab_size, 32, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss="binary_crossentropy",
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy())

    return model


def cnn_builder():
    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(vocab_size, 32, mask_zero=True),
        tf.keras.layers.Conv1D(32, 3, padding='valid', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(
                0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss="binary_crossentropy",
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy())
    return model


def rnn_builder():
    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(vocab_size, 32, mask_zero=True),
        tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2,
                             kernel_regularizer=tf.keras.regularizers.l2(
                                     0.01), recurrent_regularizer=tf.keras.regularizers.l2(0.01),
                             bias_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss="binary_crossentropy",
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy())
    return model


# main

# load data
raw_train_ds: tf.data.Dataset = text_dataset_from_directory('aclImdb/train',
                                                            batch_size=batch_size,
                                                            validation_split=0.2,
                                                            subset='training',
                                                            seed=42)

raw_val_ds: tf.data.Dataset = tf.keras.utils.text_dataset_from_directory('aclImdb/train',
                                                                         batch_size=batch_size,
                                                                         validation_split=0.2,
                                                                         subset='validation',
                                                                         seed=42)

raw_test_ds: tf.data.Dataset = tf.keras.utils.text_dataset_from_directory('aclImdb/test',
                                                                          batch_size=batch_size)

# vectorize layer
vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        output_mode='int')

# adapt to train data
vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))
vocab_size = vectorize_layer.vocabulary_size()

stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', patience=5)

model_nn = nn_builder()
model_nn.summary()

model_cnn = cnn_builder()
model_cnn.summary()

model_rnn = rnn_builder()
model_rnn.summary()

early_stopping = EarlyStopping(
        min_delta=0.005, mode='max', monitor='val_binary_accuracy', patience=2)
callback = [early_stopping]

epochs = 10
model_nn.fit(
        raw_train_ds,
        validation_data=raw_val_ds,
        epochs=epochs,
        callbacks=callback)

model_cnn.fit(
        raw_train_ds,
        validation_data=raw_val_ds,
        epochs=epochs,
        callbacks=callback)

model_rnn.fit(
        raw_train_ds,
        validation_data=raw_val_ds,
        epochs=epochs,
        callbacks=callback)

_, accuracy1 = model_nn.evaluate(raw_test_ds)

_, accuracy2 = model_cnn.evaluate(raw_test_ds)

_, accuracy3 = model_rnn.evaluate(raw_test_ds)

# baseline
train_pos = raw_train_ds.unbatch().map(lambda x, y: y).reduce(tf.constant(0), lambda x, y: x + y).numpy()

train_neg = raw_train_ds.unbatch().map(lambda x, y: y).reduce(tf.constant(0), lambda x, y: x + (1 - y)).numpy()

prediction = 1 if train_pos > train_neg else 0

num_test = raw_test_ds.unbatch().map(lambda x, y: y).reduce(tf.constant(0), lambda x, _: x + 1).numpy()
accuracy0 = raw_test_ds.unbatch().map(lambda x, y: y).reduce(tf.constant(0),
                                                             lambda x, y: x + int(y == 1)).numpy() / num_test

print('Baseline (majority class classifier)')
print("Accuracy: ", accuracy0)

print('NN')
print("Accuracy: ", accuracy1)

print('CNN')
print("Accuracy: ", accuracy2)

print('RNN')
print("Accuracy: ", accuracy3)
