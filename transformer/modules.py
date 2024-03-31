import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super(PositionalEncoding, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super(SentenceEmbedding, self).__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            try:
                sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            except:
                print(self.language_to_index)
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps # just offsetting so that we don't divide by zero
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # [512] represents a "std" (learnable)
        self.beta = nn.Parameter(torch.zeros(parameters_shape)) # [512] represents a "mean" (learnable)

    def forward(self, inputs):
        dims = [-(i+1) for i in range(len(self.parameters_shape))]  # [-1]
        mean = inputs.mean(dims, keepdim=True)                      # bs x max_seq_len x 1 (keepdim so we don't loose the last dimention)
        var = ((inputs-mean) **2).mean(dim=dims, keepdim=True)      # bs x max_seq_len x 1
        std = (var + self.eps).sqrt()                               # bs x max_seq_len x 1
        y = (inputs - mean) / std                                   # bs x max_seq_len x 512
        out = self.gamma * y + self.beta                            # bs x max_seq_len x 512
        return out
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # 512 x 2024 
        self.linear2 = nn.Linear(d_ff, d_model) # 2024 x 512
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # x: [bs, max_seq_len, 512]
        x = self.linear1(x) # [bs, max_seq_len, 2024]
        x = F.relu(x)       # [bs, max_seq_len, 2024]
        x = self.dropout(x) # [bs, max_seq_len, 2024]
        x = self.linear2(x) # [bs, max_seq_len, 512]
        return x
    
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1] # 64 (because 512/8=64)
    # k.T would just reverse the shape, but we want to transpose the last two dimentions, so use .transpose(-1, -2)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k) # scaling will keep the var somewhat close to var(q) (and var(k)) to keep gradient step stable
    if mask is not None: # if encoder, mask is None, if decoder, masking future tokens
        # we needed to permute from [B, num_heads, max_seq_len, max_seq_len] to [num_heads, B, max_seq_len, max_seq_len]
        # becaue mask is [B, max_seq_len, max_seq_len] and broadcasting (addition) happens on last 3 dimensions
        scaled = scaled.permute(1, 0, 2, 3) + mask # elementwise addition
        scaled = scaled.permute(1, 0, 2, 3) 
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    """With scaled dot product attention"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model                  # 512
        self.num_heads = num_heads              # 8
        self.head_dim = d_model // num_heads    # 64
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        self.qkv_layer = nn.Linear(d_model, 3*d_model) # 512 x 1536 (because 3 * 512)
        self.linear_layer = nn.Linear(d_model, d_model) # 522 x 512

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size() # batch_size x max_sequence_length x 512
        qkv = self.qkv_layer(x)                         # batch_size x max_sequence_length x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3*self.head_dim) # batch_size x max_sequence_length x 8 x 192 (because 3 * 64)
        qkv = qkv.permute(0, 2, 1, 3) # batch_size x 8 x max_sequence_length x 192
        q, k, v = qkv.chunk(3, dim=-1) # tuple of 3 tensors
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out
    

class MultiHeadCrossAttention(nn.Module):
    """When Q is coming from the decoder and K, V are coming from the encoder"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model                  # 512
        self.num_heads = num_heads              # 8
        self.head_dim = d_model // num_heads    # 64
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        self.kv_layer = nn.Linear(d_model, 2*d_model) # 512 x 1024 (because 2 * 512)
        self.q_layer = nn.Linear(d_model, d_model)    # 512 x 512
        self.linear_layer = nn.Linear(d_model, d_model) # 512 x 512

    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size() # bs x max_seq_len x 512
        kv = self.kv_layer(x)                           # bs x max_seq_len x 1024
        q = self.q_layer(y)                             # bs x max_seq_len x 512
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2*self.head_dim) # bs x max_seq_len x 8 x 128
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)     # bs x max_seq_len x 8 x 64
        kv = kv.permute(0, 2, 1, 3)                     # bs x 8 x max_seq_len x 128
        q = q.permute(0, 2, 1, 3)                       # bs x 8 x max_seq_len x 64
        k, v = kv.chunk(2, dim=-1)                      # tuple of 2 tensors of shape bs x 8 x max_seq_len x 64
        values, attention = scaled_dot_product(q, k, v) # values: bs x 8 x max_seq_len x 64. We don't need mask here
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model) # concatentate heads: bs x max_seq_len x 512
        out = self.linear_layer(values) # bs x max_seq_len x 512
        return out