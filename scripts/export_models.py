"""Export trained model artifacts to models/ for use by app/app.py.

Run once before starting the Gradio app:
    uv run python scripts/export_models.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import config
from utils.modeling import pseudo_label_loop

MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def export_text_unsupervised():
    print("=== Exporting text unsupervised (MiniLM + KMeans) ===")
    emb = np.load(config.CACHE_DIR / "minilm_train_full_summary.npy")

    reducer = umap.UMAP(n_components=50, metric="cosine", random_state=config.SEED)
    emb_reduced = reducer.fit_transform(emb)
    with open(MODELS_DIR / "text_umap.pkl", "wb") as f:
        pickle.dump(reducer, f)
    print(f"  UMAP saved: {emb_reduced.shape}")

    km = KMeans(n_clusters=config.NUM_CLASSES, random_state=config.SEED, n_init=10)
    km.fit(emb_reduced)
    with open(MODELS_DIR / "text_kmeans.pkl", "wb") as f:
        pickle.dump(km, f)
    print(f"  KMeans saved: {config.NUM_CLASSES} clusters")

    clusters_csv = config.RESULTS_DIR / "clusters_minilm_kmeans.csv"
    df = pd.read_csv(clusters_csv)
    cluster_names = {str(int(row["cluster"])): row["majority_true_label"]
                     for _, row in df.iterrows()}
    with open(MODELS_DIR / "text_cluster_names.json", "w") as f:
        json.dump(cluster_names, f, indent=2)
    print(f"  Cluster names saved: {len(cluster_names)} entries")


def export_text_semisupervised():
    print("=== Exporting text semi-supervised (DistilBERT pseudo-labeling) ===")
    labeled_df = pd.read_parquet(config.PROCESSED_DIR / "labeled.parquet")
    unlabeled_df = pd.read_parquet(config.PROCESSED_DIR / "unlabeled.parquet")

    model, tokenizer, current_labeled, _ = pseudo_label_loop(
        labeled_df, unlabeled_df,
        model_name=config.CLASSIFIER_MODEL_NAME,
        confidence_threshold=0.50,
        epochs=3,
        target_coverage=0.98,
        max_iterations=10,
    )
    save_dir = MODELS_DIR / "text_distilbert"
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"  DistilBERT saved to {save_dir}")


def export_image_unsupervised():
    print("=== Exporting image unsupervised (CLIP + DEC) ===")
    import torch
    from utils.dec import train_dec, _build_encoder
    from utils.image_config import (
        IMAGE_EMBEDDINGS_PATH, IMAGE_MANIFEST_PATH,
        IMAGE_CLASS_NAMES, IMAGE_NUM_CLASSES, IMAGE_SEED,
    )

    embeddings = np.load(IMAGE_EMBEDDINGS_PATH)   # (5500, 512)
    input_dim = embeddings.shape[1]
    latent_dim = 64

    labels, encoder, centers = train_dec(
        embeddings,
        n_clusters=IMAGE_NUM_CLASSES,
        latent_dim=latent_dim,
        pretrain_epochs=50,
        dec_epochs=150,
        batch_size=256,
        lr_pretrain=1e-3,
        lr_dec=1e-4,
        update_interval=5,
        tol=1e-3,
        seed=IMAGE_SEED,
    )

    torch.save(encoder.state_dict(), MODELS_DIR / "image_dec_encoder.pt")
    with open(MODELS_DIR / "image_dec_arch.json", "w") as f:
        json.dump({"input_dim": input_dim, "latent_dim": latent_dim}, f)
    np.save(MODELS_DIR / "image_dec_centers.npy", centers)
    print(f"  DEC encoder + centers saved (input_dim={input_dim}, latent_dim={latent_dim})")

    # Derive cluster→name from saved CSV (same seed → same cluster assignments)
    dec_df = pd.read_csv(REPO_ROOT / "results" / "image_full_labels_unsup_dec.csv")
    dec_df["dec_cluster"] = labels
    cluster_to_name = {}
    for cid in range(IMAGE_NUM_CLASSES):
        subset = dec_df.loc[dec_df["dec_cluster"] == cid, "true_label"]
        cluster_to_name[str(cid)] = subset.mode().iloc[0] if len(subset) > 0 else IMAGE_CLASS_NAMES[cid]
    with open(MODELS_DIR / "image_dec_cluster_names.json", "w") as f:
        json.dump(cluster_to_name, f, indent=2)
    print(f"  DEC cluster names saved: {cluster_to_name}")


def export_image_semisupervised():
    print("=== Exporting image semi-supervised (CLIP + MLP config A) ===")
    from utils.image_config import (
        IMAGE_EMBEDDINGS_PATH, IMAGE_MANIFEST_PATH,
        IMAGE_CLASS_NAMES, IMAGE_NUM_CLASSES, IMAGE_SEED,
    )
    from utils.image_pseudo_labeling import pseudo_label_loop as image_pseudo_label_loop, train_mlp

    embeddings = np.load(IMAGE_EMBEDDINGS_PATH)
    manifest = pd.read_csv(IMAGE_MANIFEST_PATH)
    seed_mask = manifest["is_seed"].to_numpy().astype(bool)
    seed_labels = manifest.loc[seed_mask, "label_idx"].to_numpy()

    pred_labels, _ = image_pseudo_label_loop(
        embeddings=embeddings,
        seed_mask=seed_mask,
        seed_labels=seed_labels,
        hidden_sizes=(256, 128),
        threshold=0.9,
        max_iters=10,
        seed=IMAGE_SEED,
        n_classes=IMAGE_NUM_CLASSES,
    )

    labeled_mask = pred_labels != -1
    final_clf = train_mlp(
        embeddings[labeled_mask], pred_labels[labeled_mask],
        hidden_sizes=(256, 128), seed=IMAGE_SEED,
    )
    with open(MODELS_DIR / "image_mlp.pkl", "wb") as f:
        pickle.dump(final_clf, f)
    with open(MODELS_DIR / "image_mlp_label_names.json", "w") as f:
        json.dump(IMAGE_CLASS_NAMES, f, indent=2)
    print(f"  MLP + label names saved — coverage: {labeled_mask.mean():.1%}")


if __name__ == "__main__":
    export_text_unsupervised()
    export_text_semisupervised()
    export_image_unsupervised()
    export_image_semisupervised()
    print("\nAll models exported to models/")
