
import tensorflow as tf
from tensorflow import keras
from keras import layers


# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Set random seed
tf.random.set_seed(1337)

#Loading data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# Create character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = tf.constant(encode(text), dtype=tf.int64)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data loading into batches
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = tf.random.uniform(shape=(batch_size,), maxval=len(data_split) - block_size, dtype=tf.int32)
    x = tf.stack([data_split[i:i+block_size] for i in ix])
    y = tf.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x, y


# Calculating loss of the model
def estimate_loss(model):
    out = {}
    model.trainable = False
    for split in ['train', 'val']:
        losses = tf.TensorArray(tf.float32, size=eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses = losses.write(k, loss)
        out[split] = losses.stack().numpy().mean()
    model.trainable = True
    return out

