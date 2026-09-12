"""
Deep Embedded Clustering (DEC) — Xie et al., 2016.

Adapted for pre-trained dense embeddings (MiniLM / RoBERTa) as input:
skips the raw-input-to-feature-extractor step and directly learns a
compressed latent space (MLP encoder, 2-layer) plus cluster centers.

Entry point: train_dec(embeddings, n_clusters, ...) -> np.ndarray of int labels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def _build_encoder(input_dim: int, latent_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, latent_dim),
    )


def _build_decoder(input_dim: int, latent_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, input_dim),
    )


# ---------------------------------------------------------------------------
# Soft assignment and target distribution
# ---------------------------------------------------------------------------

def _soft_assignment(z: torch.Tensor, centers: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Student's t-distribution soft assignment q_ij.

    Args:
        z: (N, latent_dim) latent representations
        centers: (K, latent_dim) cluster center parameters
        alpha: degrees of freedom (1 by default as in Xie et al.)

    Returns:
        q: (N, K) soft assignments, each row sums to 1
    """
    diff = z.unsqueeze(1) - centers.unsqueeze(0)          # (N, K, D)
    dist_sq = (diff ** 2).sum(dim=2)                        # (N, K)
    q = (1.0 + dist_sq / alpha) ** (-(alpha + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True)


def _target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    Sharpened target distribution p_ij.

    p_ij = (q_ij^2 / sum_i q_ij) / (sum_j q_ij^2 / sum_i q_ij)
    """
    numerator = q ** 2 / q.sum(dim=0, keepdim=True)        # (N, K)
    return numerator / numerator.sum(dim=1, keepdim=True)   # (N, K)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _pretrain_autoencoder(
    X: torch.Tensor,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> nn.Sequential:
    """Train an MLP autoencoder on X; return the encoder half."""
    torch.manual_seed(seed)
    input_dim = X.shape[1]
    encoder = _build_encoder(input_dim, latent_dim)
    decoder = _build_decoder(input_dim, latent_dim)
    optim = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    encoder.train()
    decoder.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_seen = 0
        for (xb,) in loader:
            z = encoder(xb)
            recon = decoder(z)
            loss = F.mse_loss(recon, xb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(xb)
            n_seen += len(xb)
        if (epoch + 1) % 10 == 0:
            print(f"  [DEC pretrain] epoch {epoch + 1}/{epochs}  "
                  f"recon_loss={epoch_loss / n_seen:.5f}")
    return encoder


def train_dec(
    embeddings: np.ndarray,
    n_clusters: int,
    latent_dim: int = 64,
    pretrain_epochs: int = 50,
    dec_epochs: int = 150,
    batch_size: int = 256,
    lr_pretrain: float = 1e-3,
    lr_dec: float = 1e-4,
    update_interval: int = 5,
    tol: float = 1e-3,
    seed: int = 42,
) -> np.ndarray:
    """
    Train Deep Embedded Clustering on pre-computed dense embeddings.

    Pipeline:
      1. Pretrain MLP autoencoder on `embeddings` with MSE reconstruction loss.
      2. Run KMeans on the encoder's latent space to initialise K cluster centers.
      3. Jointly refine encoder weights and cluster centers by minimising
         KL(P || Q) where Q is the soft assignment and P is the sharpened
         target distribution.
      4. Return hard cluster labels (argmax of final Q) as an int array.

    Args:
        embeddings: (N, D) float32 array of pre-trained embedding vectors.
        n_clusters: number of clusters k (must set explicitly; DEC does not
                    discover k automatically).
        latent_dim: bottleneck size for the encoder MLP.
        pretrain_epochs: autoencoder pretraining epochs.
        dec_epochs: maximum DEC refinement epochs.
        batch_size: mini-batch size for both pretraining and refinement.
        lr_pretrain: Adam learning rate for autoencoder pretraining.
        lr_dec: Adam learning rate for DEC refinement.
        update_interval: recompute P (and check convergence) every N epochs.
        tol: stop early if fraction of label changes < tol.
        seed: random seed for torch, numpy, and KMeans.

    Returns:
        labels: (N,) int32 array of cluster assignments in [0, n_clusters).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.tensor(embeddings, dtype=torch.float32)

    # 1. Pretrain encoder via autoencoder
    print(f"[DEC] Pretraining autoencoder ({pretrain_epochs} epochs)...")
    encoder = _pretrain_autoencoder(
        X, latent_dim=latent_dim, epochs=pretrain_epochs,
        batch_size=batch_size, lr=lr_pretrain, seed=seed,
    )

    # 2. KMeans initialisation in latent space
    print("[DEC] Initialising cluster centers with KMeans...")
    encoder.eval()
    with torch.no_grad():
        Z_init = encoder(X).numpy()
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    km.fit(Z_init)
    centers = nn.Parameter(
        torch.tensor(km.cluster_centers_, dtype=torch.float32)
    )

    # 3. DEC refinement: KL(P || Q)
    print(f"[DEC] Refinement (up to {dec_epochs} epochs, tol={tol})...")
    optim_dec = Adam(list(encoder.parameters()) + [centers], lr=lr_dec)
    loader_ordered = DataLoader(
        TensorDataset(X), batch_size=batch_size, shuffle=False
    )

    prev_labels: np.ndarray | None = None
    final_labels: np.ndarray = km.labels_.astype(np.int32)

    for epoch in range(dec_epochs):
        # Recompute P from full Q every update_interval epochs
        if epoch % update_interval == 0:
            encoder.eval()
            with torch.no_grad():
                q_all = _soft_assignment(encoder(X), centers)
                p_all = _target_distribution(q_all).detach()
            curr_labels = q_all.argmax(dim=1).numpy().astype(np.int32)
            final_labels = curr_labels

            if prev_labels is not None:
                delta = float((curr_labels != prev_labels).mean())
                print(f"  [DEC] epoch {epoch:4d}  label-change delta={delta:.4f}")
                if delta < tol:
                    print(f"  [DEC] Converged (delta < {tol}) at epoch {epoch}.")
                    break
            prev_labels = curr_labels.copy()

        # Mini-batch KL divergence update
        encoder.train()
        for i, (xb,) in enumerate(loader_ordered):
            start = i * batch_size
            end = start + len(xb)
            p_b = p_all[start:end]
            q_b = _soft_assignment(encoder(xb), centers)
            # KL(P||Q): sum_j p_j * log(p_j / q_j)
            loss = F.kl_div(q_b.log(), p_b, reduction="batchmean")
            optim_dec.zero_grad()
            loss.backward()
            optim_dec.step()

    print(f"[DEC] Done. {len(np.unique(final_labels))} non-empty clusters.")
    return final_labels
