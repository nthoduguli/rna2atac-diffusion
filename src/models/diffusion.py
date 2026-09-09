"""Conditional DDPM that denoises peakVI (or other continuous ATAC-representation) embeddings
conditioned on scVI cell-state embeddings."""

from __future__ import annotations

import torch
from torch import nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer-style sinusoidal embedding of diffusion timesteps."""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=timesteps.device, dtype=torch.float32) * (9.21034 / half))
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class Denoiser(nn.Module):
    """MLP denoiser predicting the noise added to an ATAC-embedding, conditioned on
    a cell-state (scVI) embedding and the diffusion timestep."""

    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        time_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.input_proj = nn.Linear(target_dim, hidden_dim)

        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, target_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_dim))
        h = self.input_proj(x_t) + t_emb + self.cond_proj(cond)
        h = self.body(h)
        return self.output_proj(h)


class ConditionalDDPM(nn.Module):
    """Conditional DDPM over continuous ATAC-representation embeddings.

    Uses a standard Gaussian forward noising process (appropriate since the
    diffusion target is a continuous peakVI embedding, not raw counts) with a
    cosine beta schedule, and an MSE noise-prediction objective.
    """

    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        n_timesteps: int = 1000,
        x0_clip: float = 10.0,
        **denoiser_kwargs,
    ) -> None:
        super().__init__()
        self.n_timesteps = n_timesteps
        self.denoiser = Denoiser(target_dim=target_dim, cond_dim=cond_dim, **denoiser_kwargs)

        betas = self._cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        # Reparametrized-x0 clip magnitude used by `sample`, saved/loaded via the
        # module's state_dict like the schedule buffers above. Overwritten by
        # load_state_dict when loading a trained checkpoint, so the constructor
        # default here only matters for a freshly-initialized model.
        self.register_buffer("x0_clip", torch.tensor(x0_clip))

    @staticmethod
    def _cosine_beta_schedule(n_timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = n_timesteps + 1
        t = torch.linspace(0, n_timesteps, steps) / n_timesteps
        f = torch.cos((t + s) / (1 + s) * torch.pi / 2)
        f = f * f
        alphas_cumprod = f / f[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[t][:, None]
        return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise

    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Standard DDPM noise-prediction MSE loss for one training step."""
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        noise_pred = self.denoiser(x_t, t, cond)
        return torch.nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Generate ATAC-representation embeddings via the standard DDPM ancestral sampler.

        Reparametrizes each step's noise prediction into a predicted x0 and clips it to
        +/-`x0_clip` before computing the (analytic true-posterior) mean and variance for
        the next step. Without this, small per-step errors in the eps-prediction compound
        multiplicatively across the reverse chain -- with `n_timesteps` in the hundreds,
        this let the sampler's output scale blow up by orders of magnitude relative to the
        training data in practice (observed empirically: predicted values in the
        hundreds/thousands against training targets in single digits).
        """
        device = cond.device
        x = torch.randn(cond.shape[0], target_dim, device=device)
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((cond.shape[0],), t, device=device, dtype=torch.long)
            alpha_bar_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            alpha_t = 1.0 - beta_t

            noise_pred = self.denoiser(x, t_batch, cond)
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-self.x0_clip, self.x0_clip)

            if t > 0:
                alpha_bar_prev = self.alphas_cumprod[t - 1]
                posterior_mean = (
                    alpha_bar_prev.sqrt() * beta_t / (1 - alpha_bar_t) * x0_pred
                    + alpha_t.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x
                )
                posterior_var = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t
                noise = torch.randn_like(x)
                x = posterior_mean + posterior_var.sqrt() * noise
            else:
                x = x0_pred
        return x
