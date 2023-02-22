import torch
from torch import nn


class DemoModel(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()
        self.encoder = nn.Linear(num_features, 512)
        self.transformer = nn.Transformer(512)
        self.decoder = nn.Linear(512, num_features)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        encoded_frame = self.encoder(frame)
        hidden_states = self.transformer(
            encoded_frame,
            torch.zeros_like(encoded_frame, device=encoded_frame.device),
        )
        result_frame = self.decoder(hidden_states)
        return result_frame
