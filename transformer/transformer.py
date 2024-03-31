import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden,        # all feed forward layers in transformer
                 num_heads,         # all multihead attention layers in transformer
                 drop_prob, 
                 num_layers,        # number of encoder and decoder layers
                 max_sequence_length,
                 other_vocab_size,  # all posible caracters in the other language
                 english_to_index,  # dictionary to convert english characters to index
                 other_to_index,    # dictionary to convert other language characters to index
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, 
                               START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, other_to_index,
                               START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, other_vocab_size) # for output
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, 
                x, # [B, max_seq_len] english sentence
                y, # [B, max_seq_len] other language sentence
                encoder_self_attention_mask=None, # [B, max_seq_len, max_seq_len] padding mask
                decoder_self_attention_mask=None, # [B, max_seq_len, max_seq_len] padding mask and look ahead mask
                decoder_cross_attention_mask=None, # [B, max_seq_len, max_seq_len] padding mask
                enc_start_token=False, # True if we want to start the decoder with the start token
                enc_end_token=False,
                dec_start_token=False,
                dec_end_token=False):
        x = self.encoder(x, encoder_self_attention_mask, enc_start_token, enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, dec_start_token, dec_end_token)
        out = self.linear(out)
        return out

