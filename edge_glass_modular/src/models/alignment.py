"""Multimodal alignment model."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass

from encoders import VisionEncoder, AudioEncoder, TextEncoder
from config import ExperimentConfig
from .losses import AlignmentLoss, TriModalAlignmentLoss


@dataclass
class AlignmentModelOutput:
    """Output from alignment model."""

    loss: Optional[torch.Tensor] = None
    losses: Optional[Dict[str, torch.Tensor]] = None
    vision_emb: Optional[torch.Tensor] = None
    audio_emb: Optional[torch.Tensor] = None
    text_emb: Optional[torch.Tensor] = None


class MultimodalAlignmentModel(nn.Module):
    """Multimodal alignment model.

    This is the main model class that combines:
    - Encoders (vision, audio, text)
    - Alignment losses

    Args:
        config: ExperimentConfig with all model settings
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()

        self.config = config

        # Initialize encoders
        self.vision_encoder = None
        self.audio_encoder = None
        self.text_encoder = None

        if config.vision_encoder is not None:
            self.vision_encoder = VisionEncoder(
                model_name=config.vision_encoder.model_name,
                projection_dim=config.vision_encoder.projection_dim,
                freeze=config.vision_encoder.freeze,
                use_perceiver=config.vision_encoder.use_perceiver,
                perceiver_num_latents=config.vision_encoder.perceiver_num_latents,
                perceiver_latent_dim=config.vision_encoder.perceiver_latent_dim,
                perceiver_num_layers=config.vision_encoder.perceiver_num_layers,
                perceiver_num_heads=config.vision_encoder.perceiver_num_heads,
                use_mrl=config.vision_encoder.use_mrl,
                mrl_dimensions=config.vision_encoder.mrl_dimensions,
            )

        if config.audio_encoder is not None:
            self.audio_encoder = AudioEncoder(
                model_name=config.audio_encoder.model_name,
                projection_dim=config.audio_encoder.projection_dim,
                freeze=config.audio_encoder.freeze,
                use_perceiver=config.audio_encoder.use_perceiver,
                perceiver_num_latents=config.audio_encoder.perceiver_num_latents,
                perceiver_latent_dim=config.audio_encoder.perceiver_latent_dim,
                perceiver_num_layers=config.audio_encoder.perceiver_num_layers,
                perceiver_num_heads=config.audio_encoder.perceiver_num_heads,
                use_mrl=config.audio_encoder.use_mrl,
                mrl_dimensions=config.audio_encoder.mrl_dimensions,
            )

        if config.text_encoder is not None:
            self.text_encoder = TextEncoder(
                model_name=config.text_encoder.model_name,
                projection_dim=config.text_encoder.projection_dim,
                freeze=config.text_encoder.freeze,
                use_mrl=config.text_encoder.use_mrl,
                mrl_dimensions=config.text_encoder.mrl_dimensions,
            )

        active_encoders = sum([
            self.vision_encoder is not None,
            self.audio_encoder is not None,
            self.text_encoder is not None,
        ])

        # Initialize loss functions
        if active_encoders == 3:
            # Tri-modal loss
            self.alignment_loss_fn = TriModalAlignmentLoss(
                vision_text_weight=1.0,
                audio_text_weight=1.0,
                vision_audio_weight=0.5,
                mrl_weight=config.optimization.mrl_loss_weight,
                temperature=0.07,
            )
        else:
            # Bi-modal loss
            self.alignment_loss_fn = AlignmentLoss(
                contrastive_weight=config.optimization.contrastive_loss_weight,
                mrl_weight=config.optimization.mrl_loss_weight,
                temperature=0.07,
            )

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        texts: Optional[Any] = None,
        return_embeddings: bool = False,
    ) -> AlignmentModelOutput:
        """Forward pass through alignment model.

        Args:
            images: Input images (batch_size, 3, H, W)
            audio_features: Input audio features (batch_size, n_mels, time)
            texts: Text strings or embeddings
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            AlignmentModelOutput with losses and embeddings
        """
        # Encode modalities
        vision_output = None
        audio_output = None
        text_output = None

        if images is not None and self.vision_encoder is not None:
            vision_output = self.vision_encoder(images)

        if audio_features is not None and self.audio_encoder is not None:
            audio_output = self.audio_encoder(audio_features)

        if texts is not None and self.text_encoder is not None:
            text_output = self.text_encoder(texts)

        # Get pooled embeddings
        vision_emb = vision_output.pooled if vision_output is not None else None
        audio_emb = audio_output.pooled if audio_output is not None else None
        text_emb = text_output.pooled if text_output is not None else None

        # Compute alignment losses
        alignment_losses = None
        if self.training or return_embeddings:
            if hasattr(self.alignment_loss_fn, "forward"):
                # Tri-modal case
                if vision_emb is not None and audio_emb is not None and text_emb is not None:
                    alignment_losses = self.alignment_loss_fn(
                        vision_emb=vision_emb,
                        audio_emb=audio_emb,
                        text_emb=text_emb,
                        vision_mrl=vision_output.mrl_embeddings if vision_output else None,
                        audio_mrl=audio_output.mrl_embeddings if audio_output else None,
                        text_mrl=text_output.mrl_embeddings if text_output else None,
                    )
                # Bi-modal cases
                elif vision_emb is not None and text_emb is not None:
                    alignment_losses = self.alignment_loss_fn(
                        embeddings_a=vision_emb,
                        embeddings_b=text_emb,
                        embeddings_a_mrl=vision_output.mrl_embeddings if vision_output else None,
                        embeddings_b_mrl=text_output.mrl_embeddings if text_output else None,
                    )
                elif audio_emb is not None and text_emb is not None:
                    alignment_losses = self.alignment_loss_fn(
                        embeddings_a=audio_emb,
                        embeddings_b=text_emb,
                        embeddings_a_mrl=audio_output.mrl_embeddings if audio_output else None,
                        embeddings_b_mrl=text_output.mrl_embeddings if text_output else None,
                    )

        # Combine losses
        total_loss = None
        if alignment_losses is not None:
            total_loss = alignment_losses.get("total_loss", 0)


        return AlignmentModelOutput(
            loss=total_loss,
            losses=alignment_losses,
            vision_emb=vision_emb if return_embeddings else None,
            audio_emb=audio_emb if return_embeddings else None,
            text_emb=text_emb if return_embeddings else None,
        )

    @property
    def num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_total_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def print_parameter_counts(self):
        """Print parameter counts for each component."""
        print(f"{'Component':<30} {'Trainable':>15} {'Total':>15}")
        print("-" * 60)

        if self.vision_encoder is not None:
            print(f"{'Vision Encoder':<30} {self.vision_encoder.num_parameters:>15,} "
                  f"{self.vision_encoder.num_total_parameters:>15,}")

        if self.audio_encoder is not None:
            print(f"{'Audio Encoder':<30} {self.audio_encoder.num_parameters:>15,} "
                  f"{self.audio_encoder.num_total_parameters:>15,}")

        if self.text_encoder is not None:
            print(f"{'Text Encoder':<30} {self.text_encoder.num_parameters:>15,} "
                  f"{self.text_encoder.num_total_parameters:>15,}")

        print("-" * 60)
        print(f"{'TOTAL':<30} {self.num_trainable_parameters:>15,} {self.num_total_parameters:>15,}")
