import numpy as np


class Embedding:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings = np.random.randn(input_dim, output_dim) * 0.1
        self.shape = self.embeddings.shape

    def forward(self, x):
        return self.embeddings[x]

    def __call__(self, x):
        return self.forward(x)



vocab_sz = 1000
context_sz = 32
embedding = Embedding(input_dim=vocab_sz, output_dim=context_sz)
print("Embedding.shape:", embedding.shape)

input_tokens = np.array([[1, 5, 10], [7, 2, 0]]) # (2, 3) = (batch, token)

embedded_output = embedding(input_tokens)

print("Shape:", embedded_output.shape)          # (2, 3, 32) = (batch, token, output=context_sz)
print(embedding.embeddings[1])                  # Embedding of token=1
print("Sample vector:", embedded_output[0, 0])  # Embedding of first sequence of first token(embeddings[input_tokens[0,0]=1])
