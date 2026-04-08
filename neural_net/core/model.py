"""
Neural network model for compression performance prediction.

Architecture (shared variant, default):
  input → first_layer_dim → [hidden_dim × (n-2)] → last_layer_dim → output

The wider first and last hidden layers give the network extra capacity to
model size-dependent effects at the input boundary and to compose richer
pre-output representations, while the narrower middle layers act as a
regularising bottleneck.

Predicts: compression_time, decompression_time, ratio, psnr
"""

import torch
import torch.nn as nn
from typing import Optional


class CompressionPredictor(nn.Module):
    """
    Multi-output regression model.

    Given (algorithm_onehot, quantization, shuffle, error_bound,
           data_size, entropy, mad, second_derivative)
    predicts (compression_time, decompression_time, ratio, psnr).

    Args:
        first_layer_dim: Width of the first hidden layer. Defaults to
            ``hidden_dim`` when ``None`` (uniform-width, backward-compatible).
        last_layer_dim:  Width of the last hidden layer before the output
            projection. Defaults to ``hidden_dim`` when ``None``.

    Shape for ``shared`` variant with ``num_hidden_layers=4``,
    ``first_layer_dim=256``, ``hidden_dim=128``, ``last_layer_dim=256``::

        input → 256 → ReLU → 128 → ReLU → 128 → ReLU → 256 → ReLU → output
    """

    def __init__(self, input_dim: int = 15, hidden_dim: int = 128, output_dim: int = 4,
                 model_variant: str = "shared", num_hidden_layers: int = 2,
                 head_hidden_dim: int = 64,
                 first_layer_dim: Optional[int] = None,
                 last_layer_dim: Optional[int] = None):
        super().__init__()
        self.model_variant = model_variant
        self.output_dim = output_dim

        _first = first_layer_dim if first_layer_dim is not None else hidden_dim
        _last  = last_layer_dim  if last_layer_dim  is not None else hidden_dim

        if model_variant == "shared":
            if num_hidden_layers < 1:
                raise ValueError("num_hidden_layers must be >= 1")

            if num_hidden_layers == 1:
                # Single hidden layer: use the first-layer width
                layers = [nn.Linear(input_dim, _first), nn.ReLU(),
                          nn.Linear(_first, output_dim)]
            elif num_hidden_layers == 2:
                # Only a first and a last hidden layer
                layers = [nn.Linear(input_dim, _first), nn.ReLU(),
                          nn.Linear(_first, _last),  nn.ReLU(),
                          nn.Linear(_last, output_dim)]
            else:
                # Wide first layer
                layers = [nn.Linear(input_dim, _first), nn.ReLU(),
                          nn.Linear(_first, hidden_dim), nn.ReLU()]
                # Uniform-width middle layers
                for _ in range(num_hidden_layers - 3):
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
                # Wide last hidden layer
                layers.extend([nn.Linear(hidden_dim, _last), nn.ReLU(),
                                nn.Linear(_last, output_dim)])

            self.net = nn.Sequential(*layers)
            self.trunk = None
            self.heads = None

        elif model_variant == "split_heads":
            # Shared trunk: wide first layer, then uniform hidden_dim.
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, _first),
                nn.ReLU(),
                nn.Linear(_first, hidden_dim),
                nn.ReLU(),
            )
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(head_hidden_dim, 1),
                )
                for _ in range(output_dim)
            ])
            self.net = None
        else:
            raise ValueError(f"Unsupported model_variant={model_variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_variant == "shared":
            return self.net(x)
        h = self.trunk(x)
        outs = [head(h) for head in self.heads]
        return torch.cat(outs, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
