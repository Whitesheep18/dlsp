import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import MultiHeadAttention, LayerNormalization, PositionwiseFeedForward, MultiHeadCrossAttention, SentenceEmbedding, PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads) # AKA cross attention
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        # we keep [bs, max_seq_len, 512] as the shape of x throughout
        residual_y = y                                  
        y = self.self_attention(y, mask=self_attention_mask)   
        y = self.dropout1(y)                           
        y = self.norm1(y + residual_y)
        residual_y = y
        y = self.encoder_decoder_attention(y, x, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.norm2(y + residual_y)
        residual_y = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y + residual_y)
        return y


class SequentialDecoder(nn.Sequential):
    """Warapper for sequential so that we can pass more than one parameter to forward method."""
    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask) # new current token [bs, max_sq_len, 512]
        return y
    

class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        # x: source sequence [bs, max_sq_len, 512]
        # y: target sequence [bs, max_sq_len, 512]
        # mask: look ahead mask [bs, max_sq_len, max_sq_len]
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y