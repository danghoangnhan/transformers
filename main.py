import torch

from Transformer import Transformer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(
        [
            [1, 5, 6, 4, 3, 9, 5, 2, 0],
            [1, 8, 7, 3, 4, 5, 6, 7, 2]
        ]
    ).to(device)
    target = torch.tensor(
        [
            [1, 7, 4, 3, 5, 9, 2, 0],
            [1, 5, 6, 2, 4, 7, 6, 2]
        ]
    ).to(device)
    source_pad_idx = 0
    target_pad_idx = 0
    source_vocab_size = 10
    target_vocab_size = 10
    model = Transformer(
        src_vocab_size=source_vocab_size,
        trg_vocab_size=target_vocab_size,
        src_pad_idx=source_pad_idx,
        trg_pad_idx=target_pad_idx,
        device=device
    ).to(device)
    out = model(x,target[:,:-1])
    print(out.shape)


