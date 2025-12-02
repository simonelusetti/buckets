import torch
import torch.nn as nn
import torch.nn.functional as F


class BucketModel(nn.Module):
    """
    Token-to-bucket assigner with trainable prototypes.

    - Each token is softly assigned to one of K buckets.
    - A sentence encoding is the weighted sum of bucket prototypes.
    """

    def __init__(
        self,
        d_model: int,
        num_buckets: int = 8,
        temperature: float = 1.0,
        prototype_ema_decay: float = 0.5,
        prototype_margin: float = 0.3,
        prototype_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_buckets = int(num_buckets)
        self.temperature = float(max(temperature, 1e-4))
        self.prototype_decay = float(prototype_ema_decay)
        self.prototype_margin = float(prototype_margin)
        self.prototype_eps = float(prototype_eps)

        # Assigner outputs alpha/beta per bucket for HardKuma gating.
        self.assigner = nn.Linear(self.d_model, 2 * self.num_buckets)
        # Prototypes are updated via EMA (not optimized by gradients).
        self.register_buffer("prototypes", torch.zeros(self.num_buckets, self.d_model))

    def _sample_gates(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce hard gates (straight-through) and soft probabilities using a
        Kumaraswamy/HardKuma sampler.
        """
        raw = self.assigner(embeddings)
        alpha_raw, beta_raw = raw.chunk(2, dim=-1)
        eps = self.prototype_eps
        alpha = F.softplus(alpha_raw) + eps
        beta = F.softplus(beta_raw) + eps

        if self.training:
            u = torch.rand_like(alpha)
            s = (1 - u.pow(1.0 / beta)).pow(1.0 / alpha)
        else:
            # Approximate mean with Beta mean; sufficient for eval.
            s = alpha / (alpha + beta)

        s = s.clamp(0.0, 1.0)
        hard = (s > 0.5).float()
        gates = hard + s - s.detach()  # straight-through estimator
        return gates, s

    def _prototype_step(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        EMA update of prototypes and prototype losses (consistency + separation).
        """
        eps = self.prototype_eps
        decay = self.prototype_decay
        margin = self.prototype_margin

        mask = attention_mask.float()
        weights = probs * mask.unsqueeze(-1)  # [B, T, K]
        weight_sum = weights.sum(dim=1).clamp_min(eps)  # [B, K]
        usage = weight_sum / mask.sum(dim=1, keepdim=True).clamp_min(eps)  # [B, K]

        # Per-bucket token centroids.
        weighted_emb = torch.einsum("btk,btd->bkd", weights, embeddings)
        factors = weighted_emb / weight_sum.unsqueeze(-1)  # [B, K, d]

        proto = self.prototypes.detach()
        diff = factors - proto.unsqueeze(0)
        cons_per = (usage * diff.pow(2).sum(dim=-1)).sum(dim=1)
        cons_loss = cons_per.mean()

        if self.training:
            with torch.no_grad():
                num = (usage.unsqueeze(-1) * factors).sum(dim=0)
                denom = usage.sum(dim=0).unsqueeze(-1).clamp_min(eps)
                update = num / denom
                self.prototypes.mul_(decay).add_(update * (1.0 - decay))

        proto_norm = proto / proto.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos = proto_norm @ proto_norm.t()
        off_diag = cos - torch.eye(cos.size(0), device=cos.device)
        sep = torch.clamp(off_diag - (1.0 - margin), min=0.0)
        sep_loss = sep.sum() / max(1, cos.numel() - cos.size(0))

        return cons_loss, sep_loss

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Args:
            embeddings: [B, T, d_model]
            attention_mask: [B, T] (1 for real tokens)
        Returns gates (hard/soft via straight-through), normalized prototypes, and prototype losses.
        """
        gates, probs = self._sample_gates(embeddings)
        proto_cons, proto_sep = self._prototype_step(embeddings, attention_mask, probs)
        prototypes = F.normalize(self.prototypes, dim=-1).detach()
        return {
            "gates": gates,
            "probs": probs,
            "prototypes": prototypes,
            "proto_cons": proto_cons,
            "proto_sep": proto_sep,
        }
