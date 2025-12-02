from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .bucket_model import BucketModel


def _entropy_loss(
    assignments: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    attn = attention_mask.float()
    tokens = attn.sum()
    entropy = -(assignments * assignments.log()).sum(dim=-1)  # [B, T]
    entropy = (entropy * attn).sum() / tokens
    return -entropy  # maximize entropy


def _balance_loss(
    global_mass: torch.Tensor,
    token_count: torch.Tensor,
    num_buckets: int
) -> torch.Tensor:
    bucket_mass = global_mass / token_count
    uniform = torch.full_like(bucket_mass, 1.0 / num_buckets)
    return torch.nn.functional.mse_loss(bucket_mass, uniform)

def _recon_loss(
    gates: torch.Tensor,
    attn: torch.Tensor,
    token_embeddings: torch.Tensor,
    sent_repr: torch.Tensor
) -> torch.Tensor:
    weights = gates * attn.unsqueeze(-1)  # [B, T, K]
    num = torch.einsum("btk,btd->bkd", weights, token_embeddings)
    denom = weights.sum(dim=1).clamp_min(1e-6)
    subsent_repr = num / denom.unsqueeze(-1)
    target_repr = subsent_repr.sum(dim=1)

    sent_norm = sent_repr.norm(dim=-1).clamp_min(1e-8)
    target_norm = target_repr.norm(dim=-1).clamp_min(1e-8)
    cos_sim = (sent_repr * target_repr).sum(dim=-1) / (sent_norm * target_norm)
    return 1.0 - cos_sim.mean()


class BucketNode:
    """
    Tree node that routes tokens into K buckets and optionally delegates
    sub-tokens to child nodes. Leaves compute the bucket losses.
    """

    def __init__(
        self,
        path: Tuple[int, ...],
        bucket_cfg,
        loss_weights: Dict[str, float],
        device: torch.device,
        sbert_encoder,
        sbert_pooler,
    ) -> None:
        self.path = path
        self.device = device
        self.loss_weights = loss_weights
        self.children: List["BucketNode"] = []

        self.model = BucketModel(
            d_model=int(bucket_cfg.d_model),
            num_buckets=int(bucket_cfg.num_buckets),
            temperature=float(bucket_cfg.temperature),
            prototype_ema_decay=float(bucket_cfg.prototype_ema_decay),
            prototype_margin=float(bucket_cfg.prototype_margin),
            prototype_eps=float(bucket_cfg.prototype_eps),
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(bucket_cfg.lr),
            weight_decay=float(bucket_cfg.weight_decay),
        )

        # shared frozen SBERT modules
        self.sbert_encoder = sbert_encoder
        self.sbert_pooler = sbert_pooler
        
    def _leaf_losses(
        self,
        attention_mask: torch.Tensor,
        gates: torch.Tensor,
        probs: torch.Tensor,
        input_ids: torch.Tensor,
        proto_cons: torch.Tensor,
        proto_sep: torch.Tensor,
    ) -> Dict[str, torch.Tensor] | None:
        attn = attention_mask.float()
        with torch.no_grad():
            features = {"input_ids": input_ids, "attention_mask": attention_mask}
            features = self.sbert_encoder(features)
            pool_out = self.sbert_pooler(features)

        token_embeddings = features["token_embeddings"]
        sent_repr = pool_out["sentence_embedding"] if isinstance(pool_out, dict) else pool_out

        # Fail-fast on SBERT outputs.
        if not torch.isfinite(token_embeddings).all():
            bad = (~torch.isfinite(token_embeddings)).nonzero(as_tuple=False)
            raise RuntimeError(f"Non-finite token embeddings at indices {bad[:5]} min={token_embeddings.min()} max={token_embeddings.max()}")
        if not torch.isfinite(sent_repr).all():
            bad = (~torch.isfinite(sent_repr)).nonzero(as_tuple=False)
            raise RuntimeError(f"Non-finite sentence embeddings at indices {bad[:5]} min={sent_repr.min()} max={sent_repr.max()}")

        recon_loss = _recon_loss(gates, attn, token_embeddings, sent_repr)
        
        token_count = attn.sum().clamp_min(1e-6)
        global_mass = (probs * attn.unsqueeze(-1)).sum(dim=(0, 1))
        entropy_loss = _entropy_loss(probs, attention_mask)
        balance_loss = _balance_loss(global_mass, token_count, self.model.num_buckets)
        loss = (
            recon_loss
            + self.loss_weights["entropy_weight"] * entropy_loss
            + self.loss_weights["balance_weight"] * balance_loss
            + self.loss_weights["prototype_pull_weight"] * proto_cons
            + self.loss_weights["prototype_repulsion_weight"] * proto_sep
        )

        components = {
            "loss": loss,
            "recon": recon_loss,
            "entropy": entropy_loss,
            "balance": balance_loss,
            "pull": proto_cons,
            "repulsion": proto_sep,
        }
        return components

    def forward_train(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> List[Dict]:
        if attention_mask.sum() == 0:
            return []

        if self.children:
            outputs = self.model(embeddings, attention_mask)
            probs = outputs["probs"].detach()
            assignments = torch.argmax(probs, dim=-1)  # [B, T]
            outputs = []
            for idx, child in enumerate(self.children):
                child_mask = (assignments == idx) & attention_mask
                if not child_mask.any():
                    continue
                child_embeddings = embeddings * child_mask.unsqueeze(-1)
                child_attention = child_mask
                child_input_ids = input_ids * child_mask
                outputs.extend(child.forward_train(child_embeddings, child_attention, child_input_ids))
            return outputs

        self.model.train()
        outputs = self.model(embeddings, attention_mask)
        gates = outputs["gates"]
        probs = outputs["probs"]
        proto_cons = outputs.get("proto_cons", probs.new_tensor(0.0))
        proto_sep = outputs.get("proto_sep", probs.new_tensor(0.0))
        components = self._leaf_losses(attention_mask, gates, probs, input_ids, proto_cons, proto_sep)
        if components is None:
            return []
        return [{"node": self, "losses": components}]

    def forward_eval(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ):
        self.model.eval()
        if attention_mask.sum() == 0:
            return []
        if self.children:
            outputs = self.model(embeddings, attention_mask)
            probs = outputs["probs"]
            assignments = torch.argmax(probs, dim=-1)
            preds = []
            for idx, child in enumerate(self.children):
                child_mask = (assignments == idx) & attention_mask
                if not child_mask.any():
                    continue
                child_embeddings = embeddings * child_mask.unsqueeze(-1)
                child_attention = child_mask
                child_input_ids = input_ids * child_mask
                preds.extend(child.forward_eval(child_embeddings, child_attention, child_input_ids))
            return preds
        # Leaf returns hard partition mask for evaluation
        outputs = self.model(embeddings, attention_mask)
        probs = outputs["probs"]
        argmax = torch.argmax(probs, dim=-1)
        gates = torch.zeros_like(probs, dtype=torch.bool)
        gates.scatter_(2, argmax.unsqueeze(-1), True)
        return [(self.path, gates, attention_mask)]


class BucketTree:
    """Lightweight tree wrapper mirroring Hydra's BranchTree but using bucket routers."""

    def __init__(
        self,
        bucket_cfg,
        loss_weights: Dict[str, float],
        device: torch.device,
        sbert_name: str,
    ) -> None:
        base = SentenceTransformer(sbert_name)
        encoder = base[0].to(device).eval()
        pooler = copy.deepcopy(base[1]).to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False
        for p in pooler.parameters():
            p.requires_grad = False

        self.bucket_cfg = bucket_cfg
        self.loss_weights = loss_weights
        self.device = device

        cfg_with_dim = copy.deepcopy(bucket_cfg)
        cfg_with_dim.d_model = bucket_cfg.d_model
        self.root = BucketNode(
            path=(),
            bucket_cfg=cfg_with_dim,
            loss_weights=loss_weights,
            device=device,
            sbert_encoder=encoder,
            sbert_pooler=pooler,
        )
        self.depth = 0

    def _expand(self, node: BucketNode):
        if not node.children:
            children = []
            for idx in range(int(self.bucket_cfg.num_buckets)):
                child_cfg = copy.deepcopy(self.bucket_cfg)
                child_cfg.d_model = self.bucket_cfg.d_model
                child = BucketNode(
                    path=node.path + (idx,),
                    bucket_cfg=child_cfg,
                    loss_weights=self.loss_weights,
                    device=self.device,
                    sbert_encoder=node.sbert_encoder,
                    sbert_pooler=node.sbert_pooler,
                )
                child.model.train()
                children.append(child)
            node.set_children(children)
        else:
            for child in node.children:
                self._expand(child)
        
    def expand(self):
        self.root.mode(train=False)
        self._expand(self.root)
        self.depth += 1

    def train_forward(self, embeddings, attention_mask, input_ids):
        return self.root.forward_train(embeddings, attention_mask, input_ids)

    def eval_forward(self, embeddings, attention_mask, input_ids):
        return self.root.forward_eval(embeddings, attention_mask, input_ids)

    def iter_nodes(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            for child in node.children:
                stack.append(child)
