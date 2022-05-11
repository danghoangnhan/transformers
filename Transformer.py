import torch
from torch import nn

from decoder.Decoder import Decoder
from encoder.Encoder import Encoder


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=256,
            num_layers=6,
            heads=8,
            device="cuda",
            forward_expansion=4,
            dropout=0,
            max_length=100
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            max_length=max_length
        )
        self.decoder = Decoder(
            trg_vocab_size=trg_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            max_length=max_length
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_source_mask(self, src):
        source_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return source_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_length = target.shape
        target_mask = torch.tril(torch.ones(target_length, target_length)).expand(
            N, 1, target_length, target_length
        )
        return target_mask.to(self.device)

    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)
        encoder_source = self.encoder(source, source_mask)
        out = self.decoder(target, encoder_source, source_mask, target_mask)
        return out
