from torch import nn

from component.SelfAttention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expanson):
        super(TransformerBlock, self).__init__()
        self.attension = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expanson * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expanson * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attension = self.attension(value, key, query, mask)
        x = self.dropout(self.norm1(attension + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out
