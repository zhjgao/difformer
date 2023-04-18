import math

from torch import nn

from improved_diffusion.gaussian_diffusion import betas_for_alpha_bar

_activations = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def build_ffn(in_dim, ffn_dim, out_dim, activation_fn="relu", drop_out=None):
    ffn = nn.Sequential(
        nn.Linear(in_dim, ffn_dim),
        _activations[activation_fn](),
        nn.Linear(ffn_dim, out_dim),
    )

    if drop_out is not None:
        ffn.append(nn.Dropout(drop_out))

    return ffn


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, rescale_factor=1.0):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        min_beta = 0.1
        max_beta = 20
        alpha_bar = lambda t: math.exp(-(max_beta - min_beta) / 2 * t ** 2 - min_beta * t)

    elif schedule_name == "cosine":
        shift = 0.008
        alpha_bar = lambda t: math.cos((t + shift) / (1 + shift) * math.pi / 2) ** 2

    elif schedule_name == 'sqrt':
        shift = 0.0001
        alpha_bar = lambda t: 1 - math.sqrt(t + shift)

    elif schedule_name == 'edm':
        rho = 7
        min_sigma = 0.002 ** (1 / rho)
        max_sigma = 80 ** (1 / rho)
        alpha_bar = lambda t: 1 / ((max_sigma + (1 - t) * (min_sigma - max_sigma)) ** (rho * 2) + 1)

    elif schedule_name == 'cdcd':
        rho = 7
        min_sigma = 1 ** (1 / rho)
        max_sigma = 300 ** (1 / rho)
        alpha_bar = lambda t: 1 / ((max_sigma + (1 - t) * (min_sigma - max_sigma)) ** (rho * 2) + 1)

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    f2 = rescale_factor ** 2
    rescaled_alpha_bar = lambda t: alpha_bar(t) / (f2 - (f2 - 1) * alpha_bar(t))
    return betas_for_alpha_bar(num_diffusion_timesteps, rescaled_alpha_bar)
