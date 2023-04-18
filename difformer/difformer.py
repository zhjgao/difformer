from random import random

import torch
from torch import nn
from torch.nn import functional as F

from fairseq import utils
from fairseq.models import register_model, register_model_architecture, transformer
from fairseq.models.nat import NATransformerModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from improved_diffusion.gaussian_diffusion import GaussianDiffusion
from improved_diffusion.respace import SpacedDiffusion, space_timesteps

from .modules import DifformerEncoder, DifformerDecoder
from .utils import get_named_beta_schedule


@register_model("difformer")
class Difformer(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

        parser.add_argument(
            "--model-dim",
            type=int, metavar="N",
            help="The dimension of the model"
        )
        parser.add_argument(
            "--latent-dim",
            type=int, metavar="N",
            help="The dimension of $z_t$"
        )

        parser.add_argument(
            "--share-project-in-dim",
            action="store_true",
            help="Share projection layers of the encoder and decoder"
        )

        parser.add_argument(
            "--diffusion-steps",
            type=int, metavar="N", default=2000,
            help="Diffusion steps"
        )

        parser.add_argument(
            "--noise-schedule",
            type=str, metavar="STR", default="sqrt",
            help="The noise schedule during training"
        )
        parser.add_argument(
            "--noise-factor",
            type=float, metavar="D", default=1.0,
            help="The noise factor during training"
        )
        parser.add_argument(
            "--rescale-factor",
            type=float, metavar="D", default=1.0,
            help="When change the noise factor, both the signal-to-noise ratio (SNR) of the \
                noise schedule and the variance of $z_t$ are changed. The rescale factor only \
                rescales the noise schedule, so that it has a equivalent SNR, but keeps the \
                variance of $z_t$ unchanged."
        )

        parser.add_argument(
            "--embed-norm",
            action="store_true",
            help="Add embedding layer normalization"
        )
        parser.add_argument(
            "--embed-norm-affine",
            action="store_true",
            help="Add elementwise affine parameters to the embedding layer normalization"
        )
        parser.add_argument(
            "--embed-norm-before-proj",
            action="store_true",
            help="Put the embedding layer normalization before the projection layers"
        )

        parser.add_argument(
            "--self-cond",
            action="store_true",
            help="Self-conditioning"
        )
        parser.add_argument(
            "--self-cond-before-proj",
            action="store_true",
            help="Concatenate self-conditioning embeddings before the projection layers"
        )

        parser.add_argument(
            "--rounding-loss",
            action="store_true",
            help="Use the rounding loss instead of the anchor loss"
        )

        parser.add_argument(
            "--rescale-timesteps",
            action="store_true",
            help="Pass floating point timesteps into the model"
        )

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.training_diffusion = GaussianDiffusion(
            betas=get_named_beta_schedule(
                args.noise_schedule,
                args.diffusion_steps,
                args.rescale_factor
            ),
            model_mean_type=None, model_var_type=None, loss_type=None
        )

        # so we have different schedules in training and decoding
        self.decoding_diffusion = SpacedDiffusion(
            space_timesteps(args.diffusion_steps, str(args.decoding_steps)),
            betas=get_named_beta_schedule(
                args.noise_schedule,
                args.diffusion_steps,
                args.decoding_rescale_factor
            ),
            model_mean_type=None, model_var_type=None, loss_type=None
        )

        self.timesteps_scale = (1000.0 / args.diffusion_steps) if args.rescale_timesteps else 1.0
        
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, project_in_dim):
        return DifformerEncoder(args, src_dict, embed_tokens, project_in_dim)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, project_in_dim, project_out_dim):
        decoder = DifformerDecoder(args, tgt_dict, embed_tokens, project_in_dim, project_out_dim)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """ Build a new model instance """

        transformer.base_architecture(args)
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = transformer.DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = transformer.DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        latent_dim = args.latent_dim
        model_dim = args.model_dim

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, latent_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, latent_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, latent_dim, args.encoder_embed_path
            )

        # projection layers
        if latent_dim != model_dim:
            encoder_project_in_dim = nn.Linear(latent_dim, model_dim, bias=False)
            decoder_project_in_dim = (
                encoder_project_in_dim if args.share_project_in_dim
                else nn.Linear(latent_dim, model_dim, bias=False)
            )
            
            decoder_project_out_dim = nn.Linear(model_dim, latent_dim, bias=False)
        
        else:
            encoder_project_in_dim = nn.Identity()
            decoder_project_in_dim = nn.Identity()
            decoder_project_out_dim = nn.Identity()

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, encoder_project_in_dim)
        decoder = cls.build_decoder(
            args, tgt_dict, decoder_embed_tokens,
            decoder_project_in_dim, decoder_project_out_dim
        )

        return cls(args, encoder, decoder)

    def forward(self, src_tokens, src_lengths, _, tgt_tokens, **kwargs):
        """ Compute training losses """

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
        mask = tgt_tokens.ne(self.pad)

        # diffusion
        z_0 = self.decoder.forward_embedding(tgt_tokens)
        t = torch.randint(0, self.args.diffusion_steps, [len(z_0)], device=z_0.device)
        model_t = t * self.timesteps_scale

        noise = torch.randn_like(z_0) * self.args.noise_factor
        z_t = self.training_diffusion.q_sample(z_0, t, noise).type_as(z_0)

        # self-conditioning
        prev_z_0_hat = torch.zeros_like(z_0)
        if self.args.self_cond and random() < 0.5:
            with torch.no_grad():
                prev_z_0_hat = self.decoder(z_t, model_t, mask, encoder_out, prev_z_0_hat)[0]
        
        z_0_hat = self.decoder(z_t, model_t, mask, encoder_out, prev_z_0_hat)[0]
        logits = self.decoder.output_layer(z_0 if self.args.rounding_loss else z_0_hat)

        return {
            "diffusion": {
                "loss": (z_0_hat - z_0)[mask].square().mean()
            },

            "word_ins": {
                "out": logits,
                "tgt": tgt_tokens,
                "mask": mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }

    def forward_decoder(self, z_t, step, mask, encoder_out, prev_z_0_hat=None, **kwargs):
        """ Sample z_{t-1} given z_t """

        # rescale timesteps
        model_t = (
            self.decoding_diffusion.timestep_map[step]
            if self.args.decoding_fixed_t is None
            else self.args.decoding_fixed_t * self.args.diffusion_steps
        ) * self.timesteps_scale
        model_t = torch.full([len(z_t)], model_t, device=z_t.device)

        # predict z_0            
        z_0_hat = self.decoder(z_t, model_t, mask, encoder_out, prev_z_0_hat)[0]

        # clamping trick
        if self.args.clamping:
            tokens = self.decoder.output_layer(z_0_hat).argmax(-1)
            z_0_hat = self.decoder.forward_embedding(tokens)

        # sample z_{t-1}
        t = torch.tensor(step, device=z_t.device)
        mean, _, log_variance = self.decoding_diffusion.q_posterior_mean_variance(z_0_hat, z_t, t)
        noise = torch.randn_like(z_t) * self.args.decoding_noise_factor

        z_t = mean + (0.5 * log_variance).exp() * noise
        z_t = z_t.type_as(z_0_hat)

        return z_t, z_0_hat

    def forward_output_layer(self, z_t, mask):
        logits, tokens = self.decoder.output_layer(z_t).max(-1)
        scores = F.log_softmax(logits, -1)
        return tokens, scores, mask

    def initialize_z_t(self, encoder_out):
        """ Sample z_T """
        # length prediction
        pred_length = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )

        max_length = pred_length.clamp_(min=2).max()
        z_t = torch.randn(
            (len(pred_length), max_length, self.args.latent_dim),
        ) * self.args.decoding_noise_factor

        return z_t, pred_length

    def regenerate_beam(self, pred_length, length_beam_size, noise_beam_size):
        pred_length = (
            pred_length[:, None, None]
            + utils.new_arange(pred_length, 1, noise_beam_size, length_beam_size).transpose(-1, -2)
            - length_beam_size // 2
        ).flatten()  # (bsz * length_beam_size * noise_beam_size)

        max_length = pred_length.clamp_(min=2).max()
        z_t = torch.randn(
            (len(pred_length), max_length, self.args.latent_dim),
        ) * self.args.decoding_noise_factor

        return z_t, pred_length


@register_model_architecture("difformer", "difformer")
def base_architecture(args):
    args.model_dim = getattr(args, "model_dim", 512)
    args.latent_dim = getattr(args, "latent_dim", 128)

    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = args.model_dim
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = args.model_dim
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )

    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.latent_dim)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_input_dim)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.share_project_in_dim = getattr(args, "share_project_in_dim", False)

    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)

    args.diffusion_steps = getattr(args, "diffusion_steps", 2000)

    args.noise_schedule = getattr(args, "noise_schedule", "linear")
    args.noise_factor = getattr(args, "noise_factor", 1.0)
    args.rescale_factor = getattr(args, "rescale_factor", 1.0)

    args.embed_norm = getattr(args, "embed_norm", False)
    args.embed_norm_affine = getattr(args, "embed_norm_affine", False)
    args.embed_norm_before_proj = getattr(args, "embed_norm_before_proj", False)

    args.self_cond = getattr(args, "self_cond", False)
    args.self_cond_before_proj = getattr(args, "self_cond_before_proj", False)

    args.rounding_loss = getattr(args, "rounding_loss", False)

    args.rescale_timesteps = getattr(args, "rescale_timesteps", False)


@register_model_architecture("difformer", "difformer_base")
def difformer_base(args):
    args.model_dim = getattr(args, "model_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)

    base_architecture(args)


@register_model_architecture("difformer", "difformer_iwslt_de_en")
def difformer_nat_iwslt_de_en(args):
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    base_architecture(args)


@register_model_architecture("transformer", "transformer_base")
def transformer_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    transformer.base_architecture(args)
