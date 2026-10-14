"""Transformer factory used by the classifier."""
from torch import nn

def transformer(dim, heads, layers):
    block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.,
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False)

