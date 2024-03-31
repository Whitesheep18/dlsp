import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import MultiHeadAttention, LayerNormalization, PositionwiseFeedForward, SentenceEmbedding, PositionalEncoding


class SequentialEncoder(nn.Sequential):
    """Warapper for sequential so that we can pass more than one parameter to forward method."""
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=ffn_hidden)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask):
        # we keep [bs, max_seq_len, 512] as the shape of x throughout
        residual_x = x                      # keep a copy of x for residual connection to add to the output of the attention layer
        x = self.attention(x, mask=mask)    # masking only padding tokens
        x = self.dropout1(x)                # apply dropout
        x = self.norm1(x + residual_x)      # add residual connection and apply layer normalization
        residual_x = x                      # keep a copy of x for residual connection to add to the output of the feed forward layer
        x = self.ffn(x)                     
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, 
                 language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super(Encoder, self).__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, num_heads=num_heads, drop_prob=drop_prob) for _ in range(num_layers)])
        
    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token) # outout will not include a start token but will include an end token
        x = self.layers(x, self_attention_mask) # list of contextually aware word embeddings
        return x