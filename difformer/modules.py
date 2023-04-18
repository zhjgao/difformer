import enum
import math
from typing import Optional

import torch
from torch import nn

from fairseq.models.nat import NATransformerDecoder
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import PositionalEmbedding

from improved_diffusion.nn import timestep_embedding

from .utils import build_ffn


class DifformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, project_in_dim=None):
        super().__init__(args, dictionary, embed_tokens)

        self.project_in_dim = project_in_dim

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                args.encoder_embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        
        x = embed = self.embed_scale * token_embedding
        x = self.project_in_dim(x)

        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed


class EmbedNormPosition(enum.Enum):
    NO_EMBED_NORM = enum.auto()
    BEFORE_PROJ = enum.auto()
    AFTER_PROJ = enum.auto()


class SelfCondPosition(enum.Enum):
    NO_SELF_COND = enum.auto()
    BEFORE_PROJ = enum.auto()
    AFTER_PROJ = enum.auto()


class DifformerDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, project_in_dim=None, project_out_dim=None):
        super().__init__(args, dictionary, embed_tokens)

        latent_dim = args.latent_dim
        model_dim = args.model_dim
        
        self.project_in_dim = project_in_dim
        self.project_out_dim = project_out_dim

        # embedding normalization
        if not args.embed_norm:
            args.embed_norm_position = EmbedNormPosition.NO_EMBED_NORM
        elif args.embed_norm_before_proj:
            args.embed_norm_position = EmbedNormPosition.BEFORE_PROJ
        else:
            args.embed_norm_position = EmbedNormPosition.AFTER_PROJ
        
        if args.embed_norm:
            self.embed_norm = nn.LayerNorm(
                latent_dim if args.embed_norm_position == EmbedNormPosition.BEFORE_PROJ
                else model_dim,
                elementwise_affine=args.embed_norm_affine
            )

        # self-conditioning
        if not args.self_cond:
            args.self_cond_position = SelfCondPosition.NO_SELF_COND
        elif args.self_cond_before_proj:
            args.self_cond_position = SelfCondPosition.BEFORE_PROJ
        else:
            args.self_cond_position = SelfCondPosition.AFTER_PROJ
        
        if args.self_cond:
            self_cond_dim = (
                latent_dim if args.self_cond_position == SelfCondPosition.BEFORE_PROJ
                else model_dim
            )

            self.self_cond_proj = build_ffn(
                self_cond_dim * 2, self_cond_dim, self_cond_dim,
                args.activation_fn, args.dropout
            )

        self.embed_time = build_ffn(model_dim, model_dim * 4, model_dim, args.activation_fn)

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.latent_dim)

    def forward_embedding(self, tokens):
        embed = self.embed_scale * self.embed_tokens(tokens)

        if self.args.embed_norm_position == EmbedNormPosition.BEFORE_PROJ:
            embed = self.embed_norm(embed)
        
        return embed

    def forward(self, z_t, t, mask, encoder_out, prev_z_0_hat=None, **kwargs):
        hidden = self.forward_hidden(z_t, t, mask, prev_z_0_hat)

        # B x T x C -> T x B x C
        hidden = hidden.transpose(0, 1)
        attn = None
        inner_states = [hidden]

        # decoder layers
        for i, layer in enumerate(self.layers):
            hidden, attn, _ = layer(
                hidden,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=~mask,
            )
            inner_states.append(hidden)

        if self.layer_norm:
            hidden = self.layer_norm(hidden)

        # T x B x C -> B x T x C
        hidden = hidden.transpose(0, 1)

        hidden = self.project_out_dim(hidden)
        return hidden, {"attn": attn, "inner_states": inner_states}

    def forward_hidden(self, z_t, t, mask, prev_z_0_hat=None):
        # self-conditioning
        if self.args.self_cond_position == SelfCondPosition.BEFORE_PROJ:
            cat_embed = torch.cat((z_t, prev_z_0_hat), -1)
            hidden = self.project_in_dim(self.self_cond_proj(cat_embed))
        
        elif self.args.self_cond_position == SelfCondPosition.AFTER_PROJ:
            z_hidden = self.project_in_dim(z_t)
            prev_hidden = self.project_in_dim(prev_z_0_hat)
            cat_hidden = torch.cat((z_hidden, prev_hidden), -1)
            hidden = self.self_cond_proj(cat_hidden)
        
        else:
            hidden = self.project_in_dim(z_t)

        # time embedding
        time_emb = self.embed_time(timestep_embedding(t, self.args.model_dim).type_as(z_t))[:, None]
        hidden = hidden + time_emb

        # position embedding
        positions = self.embed_positions(mask.long() + self.padding_idx)
        hidden = hidden + positions
        
        # embedding normalization
        if self.args.embed_norm_position == EmbedNormPosition.AFTER_PROJ:
            hidden = self.embed_norm(hidden)
        
        hidden = self.dropout_module(hidden)
        return hidden
