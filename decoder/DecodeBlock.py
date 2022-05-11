from torch import nn
from component.SelfAttention import SelfAttention
from component.Transformer import TransformerBlock


class DecoderBlock(nn.Module):
    def __init__(self,
                 embed_size,
                 heads,
                 device,
                 forward_expansion,
                 dropout):
        super(DecoderBlock, self).__init__()
        self.attension = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size=embed_size,
            heads=heads,
            dropout=dropout,
            forward_expanson=forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attension = self.attension(x, x, x, trg_mask)
        query = self.dropout(self.norm(attension + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

