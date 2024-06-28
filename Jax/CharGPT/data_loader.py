import tensorflow as tf
import numpy as np

class read_txt:
    def __init__(self, filepath="sha.txt"):
        self.filepath = filepath
        self.text = self.read_file(filepath)
        self.tokens, self.detokens = self.get_tokens()
        self.tokenized_text = np.array([self.tokens[i] for i in self.text], dtype=np.int32)
        self.information()

    def information(self):
        print(f"The file {self.filepath} is being processed. There are {len(self.tokens)} many unique tokens!")

    def get_tokens(self):
        unique_chars = sorted(set(self.text))
        tokens = {char: i for i, char in enumerate(unique_chars)}
        detokens = {i: char for char, i in tokens.items()}
        return tokens, detokens

    def read_file(self, filepath):
        with open(filepath, 'r') as file:
            return file.read()


def return_dataset(path="sha.txt",
                   context_length=128,
                   batch_size=128,
                   epochs=3,
                   train_split=0.8,
                   buffer_size=5000):
    txt_tokenized = read_txt(path)
    tokenizer, detokenizer = txt_tokenized.tokens, txt_tokenized.detokens
    
    data = txt_tokenized.tokenized_text
    total_length = len(data)
    train_length = int(total_length * train_split)
    
    train_data = data[:train_length]
    val_data = data[train_length:]

    def create_dataset(input_data, is_training=False):
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.window(context_length + 1, shift=1, stride=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda x: x.batch(context_length + 1))
        
        if is_training:
            dataset = dataset.cache()
            dataset = dataset.shuffle(buffer_size)
            dataset = dataset.repeat(epochs)
        
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_dataset(train_data, is_training=True)
    val_dataset = create_dataset(val_data)


    return train_dataset, val_dataset, tokenizer, detokenizer

"""
train_iter, val_iter, tokenizer, detokenizer = return_dataset(batch_size=128, context_length=32)

for train_batch in train_iter:
    x_train, y_train = train_batch[:, :-1], train_batch[:, 1:]
    print(x_train.shape, y_train.shape)

for val_batch in val_iter:
    x_val, y_val = val_batch[:, :-1], val_batch[:, 1:]
"""