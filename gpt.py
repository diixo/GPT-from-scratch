
import tensorflow as tf
from tensorflow import keras
from keras import layers


# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
eval_iters = 100
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


#--------------------------------------------------------------------------------------------------------
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


#--------------------------------------------------------------------------------------------------------
class Head(tf.keras.layers.Layer):
    """ one head of self-attention """

    def __init__(self, head_size):
        super(Head, self).__init__()
        self.key = tf.keras.layers.Dense(head_size, use_bias=False)
        self.query = tf.keras.layers.Dense(head_size, use_bias=False)
        self.value = tf.keras.layers.Dense(head_size, use_bias=False)

        tril = tf.linalg.band_part(tf.ones((block_size, block_size)), -1, 0)
        self.tril = tf.constant(tril)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)

        # compute attention scores ("affinities")
        wei = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])) * tf.math.rsqrt(tf.cast(k.shape[-1], tf.float32))  # (B, T, T)
        wei = tf.where(self.tril[:T, :T] == 0, float('-inf'), wei)  # (B, T, T)
        # the same as previous: wei = wei + tf.math.log(self.tril[:T, :T])
        wei = tf.nn.softmax(wei, axis=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)       # (B, T, hs)
        out = tf.matmul(wei, v) # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


#--------------------------------------------------------------------------------------------------------
class MultiHeadAttention(tf.keras.layers.Layer):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = tf.keras.layers.Dense(n_embd)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        out = tf.concat([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        return out


#--------------------------------------------------------------------------------------------------------
class FeedForward(tf.keras.layers.Layer):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super(FeedForward, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * n_embd),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(n_embd),
            tf.keras.layers.Dropout(dropout),
        ])

    def call(self, x):
        return self.net(x)


#--------------------------------------------------------------------------------------------------------
class Block(tf.keras.layers.Layer):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super(Block, self).__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


#--------------------------------------------------------------------------------------------------------
class GPTLanguageModel(tf.keras.Model):

    def __init__(self):
        super(GPTLanguageModel, self).__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = tf.keras.layers.Embedding(block_size, n_embd)
        self.blocks = tf.keras.Sequential([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.lm_head = tf.keras.layers.Dense(vocab_size, kernel_initializer='normal', bias_initializer='zeros')

    def call(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(tf.range(T, dtype=tf.float32))  # (T,C)
        x = tok_emb + pos_emb   # (B,T,C)
        x = self.blocks(x)      # (B,T,C)
        x = self.ln_f(x)        # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = tf.reshape(logits, (B * T, C))
            targets = tf.reshape(targets, (B * T,))
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(targets, logits)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = tf.nn.softmax(logits, axis=-1)  # (B, C)
            # sample from the distribution
            idx_next = tf.random.categorical(tf.math.log(probs), num_samples=1, dtype=tf.int64)  # (B, 1)
            # append sampled index to the running sequence
            idx = tf.concat([idx, idx_next], axis=1)  # (B, T+1)
        return idx


#--------------------------------------------------------------------------------------------------------
#Training the model. GPU is recommended for training.

model = GPTLanguageModel()
optimizer = tf.keras.optimizers.Adam(learning_rate)

for iter in range(max_iters):

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    with tf.GradientTape() as tape:
        logits, loss = model(xb, yb)

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        #losses = estimate_loss(model)
        print(f"...on iter={iter}: train loss={loss:.4f}")


    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


#--------------------------------------------------------------------------------------------------------
# generate from the model
context = tf.zeros((1, 1), dtype=tf.int64)
generated_sequence = model.generate(context, max_new_tokens=500).numpy()
print(decode(generated_sequence[0]))

model.save_weights('gpt_model_weights.h5')
