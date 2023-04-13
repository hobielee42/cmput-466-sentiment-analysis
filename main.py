from keras.utils import text_dataset_from_directory

raw_train_ds = text_dataset_from_directory('aclImdb/train', batch_size=32, validation_split=0.2, subset='training',
                                           seed=42)
