import open_clip
import open_clip.transformer
import torch


class LanguageTransformer(torch.nn.Module):
    """Language Transformer for CLIP.

    Args:
        transformer (Transformer): Transformer model.
        token_embedding (torch.nn.Embedding): Token embedding.
        positional_embedding (torch.nn.Parameter): Positional embedding.
        ln_final (torch.nn.LayerNorm): Layer norm.
        text_projection (torch.nn.Parameter): Text projection.
    """

    def __init__(
        self,
        model: open_clip.transformer.Transformer,
        token_embedding: torch.nn.Embedding,
        positional_embedding: torch.nn.Parameter,
        ln_final: torch.nn.LayerNorm,
        text_projection: torch.nn.Parameter,
        attn_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.transformer = model
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.ln_final = ln_final
        self.text_projection = text_projection

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            text (torch.Tensor): Text tensor.
        """
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
