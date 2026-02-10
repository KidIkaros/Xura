"""PyTorch model mirrors for Kore Mamba3-JEPA training.

These modules replicate the Rust architecture exactly so that weights
can be trained in PyTorch and exported to safetensors for Rust inference.
"""

from .mamba3 import Mamba3Layer, Mamba3Backbone, RMSNorm
from .predictor import Mamba3Predictor
from .angn import AdaptiveNeuralGate, ANGNConfig
from .y_encoder import Mamba3TextEncoder
from .y_decoder import Mamba3Decoder
from .vit import VisionEncoder
from .vljepa import Mamba3Jepa

__all__ = [
    "Mamba3Layer",
    "Mamba3Backbone",
    "RMSNorm",
    "Mamba3Predictor",
    "AdaptiveNeuralGate",
    "ANGNConfig",
    "Mamba3TextEncoder",
    "Mamba3Decoder",
    "VisionEncoder",
    "Mamba3Jepa",
]
