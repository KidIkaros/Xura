"""Mamba-3 Decoder (Y-Decoder) — PyTorch mirror of kore-vljepa's y_decoder.rs.

Takes predicted embeddings from the predictor and decodes them into text
tokens using a Mamba-3 language model with prefix conditioning.

Weight key mapping:
  y_decoder.embed_to_prefix.weight → y_decoder.embed_to_prefix.weight
  y_decoder.embed_to_prefix.bias   → y_decoder.embed_to_prefix.bias
  y_decoder.backbone.*             → y_decoder.backbone.*
  y_decoder.lm_head.weight         → y_decoder.lm_head.weight
  y_decoder.token_embedding.weight → y_decoder.token_embedding.weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba3 import Mamba3Backbone, Mamba3Config


class Mamba3Decoder(nn.Module):
    """Y-Decoder: predicted embedding → text tokens.

    Architecture:
      embedding → embed_to_prefix → (prefix_len, d_model) prefix tokens
      prefix + token_embedding(prev_tokens) → Mamba-3 backbone → lm_head → logits

    The decoder is conditioned on the predicted embedding via learned prefix
    tokens, then autoregressively generates text.

    Mirrors: kore-vljepa's Mamba3Decoder.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 32,
        vocab_size: int = 32000,
        prefix_len: int = 8,
        embed_dim: int = 1536,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.prefix_len = prefix_len
        self.embed_dim = embed_dim

        # Embedding → prefix tokens: (embed_dim) → (prefix_len * d_model)
        self.embed_to_prefix = nn.Linear(embed_dim, prefix_len * d_model)

        # Token embedding for autoregressive generation
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Mamba-3 backbone
        mamba_config = Mamba3Config(
            d_model=d_model,
            n_layer=n_layers,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            vocab_size=vocab_size,
        )
        self.backbone = Mamba3Backbone(mamba_config)

        # LM head: d_model → vocab_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        predicted_embedding: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward pass with teacher forcing.

        Args:
            predicted_embedding: (batch, embed_dim) from predictor
            target_tokens: (batch, seq_len) ground-truth token IDs

        Returns:
            (batch, seq_len, vocab_size) logits for cross-entropy loss
        """
        batch = predicted_embedding.shape[0]

        # Generate prefix from predicted embedding
        prefix = self.embed_to_prefix(predicted_embedding)          # (B, prefix_len * d_model)
        prefix = prefix.view(batch, self.prefix_len, self.d_model)  # (B, prefix_len, d_model)

        # Embed target tokens
        tok_embed = self.token_embedding(target_tokens)             # (B, seq_len, d_model)

        # Concatenate: [prefix; token_embeddings]
        hidden = torch.cat([prefix, tok_embed], dim=1)              # (B, prefix_len+seq_len, d_model)

        # Backbone
        hidden = self.backbone(hidden)

        # LM head on token positions only (skip prefix)
        token_hidden = hidden[:, self.prefix_len:]                  # (B, seq_len, d_model)
        logits = self.lm_head(token_hidden)                         # (B, seq_len, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        predicted_embedding: torch.Tensor,
        bos_token: int = 0,
        eos_token: int = 1,
        max_tokens: int = 64,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        """Autoregressive generation.

        Args:
            predicted_embedding: (batch, embed_dim) from predictor
            bos_token: Beginning-of-sequence token ID
            eos_token: End-of-sequence token ID
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = no change)

        Returns:
            List of token ID sequences, one per batch element.
        """
        batch = predicted_embedding.shape[0]
        device = predicted_embedding.device

        # Generate prefix
        prefix = self.embed_to_prefix(predicted_embedding)
        prefix = prefix.view(batch, self.prefix_len, self.d_model)

        # Start with BOS
        generated = [[bos_token] for _ in range(batch)]
        current_tokens = torch.full((batch, 1), bos_token, dtype=torch.long, device=device)
        finished = [False] * batch

        for _ in range(max_tokens):
            tok_embed = self.token_embedding(current_tokens)
            hidden = torch.cat([prefix, tok_embed], dim=1)
            hidden = self.backbone(hidden)

            # Get logits for last position
            last_logits = self.lm_head(hidden[:, -1])  # (B, vocab_size)

            if temperature > 0:
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_token = last_logits.argmax(dim=-1)

            for b in range(batch):
                if not finished[b]:
                    tok = next_token[b].item()
                    generated[b].append(tok)
                    if tok == eos_token:
                        finished[b] = True

            if all(finished):
                break

            # Append to sequence for next step
            current_tokens = torch.cat([
                current_tokens, next_token.unsqueeze(1)
            ], dim=1)

        return generated

    @classmethod
    def small(cls) -> "Mamba3Decoder":
        """Small preset matching Rust Mamba3DecoderConfig::small()."""
        return cls(
            d_model=512, n_layers=6, d_state=64,
            expand=2, headdim=32, vocab_size=32000,
            prefix_len=8, embed_dim=1536,
        )

    @classmethod
    def tiny(cls) -> "Mamba3Decoder":
        """Tiny preset for testing."""
        return cls(
            d_model=64, n_layers=2, d_state=16,
            expand=2, headdim=16, vocab_size=256,
            prefix_len=4, embed_dim=32,
        )
