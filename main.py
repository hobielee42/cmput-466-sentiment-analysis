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


def nn():
    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(vocab_size, 100, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss="binary_crossentropy",
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy())

    return model


def cnn():
    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(vocab_size, 100, mask_zero=True),
        tf.keras.layers.Conv1D(32, 3, padding='valid', activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss="binary_crossentropy",
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy())
    return model


def rnn():
    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(vocab_size, 100, mask_zero=True),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(
                0.01), recurrent_regularizer=tf.keras.regularizers.l2(0.01),
                             bias_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss="binary_crossentropy",
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy())
    return model


# main
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

small_train_ds = raw_train_ds.take(30)
small_val_ds = raw_val_ds.take(30)
small_test_ds = raw_test_ds.take(30)

vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        output_mode='int')

vectorize_layer.adapt(small_train_ds.map(lambda x, y: x))
vocab_size = vectorize_layer.vocabulary_size()

model1 = nn()
model1.summary()

model2 = cnn()
model2.summary()

model3 = rnn()
model3.summary()

early_stopping = EarlyStopping(min_delta=0.005, mode='max', monitor='val_acc', patience=2)
callback = [early_stopping]

epochs = 10
model1.fit(
        # raw_train_ds,
        # validation_data=raw_val_ds,
        small_train_ds,
        validation_data=small_val_ds,
        epochs=epochs,
        callbacks=callback)

model2.fit(
        # raw_train_ds,
        # validation_data=raw_val_ds,
        small_train_ds,
        validation_data=small_val_ds,
        epochs=epochs,
        callbacks=callback)

model3.fit(
        # raw_train_ds,
        # validation_data=raw_val_ds,
        small_train_ds,
        validation_data=small_val_ds,
        epochs=epochs,
        callbacks=callback)

loss1, accuracy1 = model1.evaluate(small_test_ds)

loss2, accuracy2 = model2.evaluate(small_test_ds)

loss3, accuracy3 = model3.evaluate(small_test_ds)

print('NN')
print("Loss: ", loss1)
print("Accuracy: ", accuracy1)

print('CNN')
print("Loss: ", loss2)
print("Accuracy: ", accuracy2)

print('RNN')
print("Loss: ", loss3)
print("Accuracy: ", accuracy3)
