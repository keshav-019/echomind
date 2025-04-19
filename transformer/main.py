import requests

import numpy as np

import os

import tensorflow as tf

data_path = "data/input.txt"

if not os.path.exists(data_path):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(response.text)

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

print("Length of text:", len(text))
print("Sample:", text[:500])

# Character vocabulary
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Encode the text
text_as_int = np.array([char2idx[c] for c in text])

vocab_size = len(vocab)
print("Vocab size:", vocab_size)

seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch and shuffle
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def get_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits /= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        return self.dense(output)



def transformer_block(d_model, num_heads, ff_dim, rate=0.1):
    inputs = tf.keras.Input(shape=(None, d_model))
    attention = MultiHeadAttention(d_model, num_heads)(inputs, inputs, inputs)
    attention = tf.keras.layers.Dropout(rate)(attention)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

    ffn_output = ffn(out1)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return tf.keras.Model(inputs=inputs, outputs=out2)



def create_model(seq_len, vocab_size, d_model=128, num_heads=4, ff_dim=512, num_layers=2):
    inputs = tf.keras.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)

    pos_encoding = get_positional_encoding(seq_len, d_model)
    x += pos_encoding[:, :seq_len, :]

    for _ in range(num_layers):
        x = transformer_block(d_model, num_heads, ff_dim)(x)

    x = tf.keras.layers.Dense(vocab_size)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)



model = create_model(seq_length, vocab_size)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.summary()

EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS)




