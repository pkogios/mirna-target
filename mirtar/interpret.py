"""What did each model learn? Figures from saved models, on validation pairs.

    python -m mirtar.interpret --split stratified --seed 0

Writes to results/<split>/interpretation/:
    m1_shap.png, m1_shap.csv     mean |SHAP| per feature and per feature group (M1)
    position_profiles.png/.csv   importance of each miRNA position for M2 (attention
                                 received from the target) and M3 (input-gradient
                                 saliency), averaged over validation positives
    m3_saliency_examples.png     saliency on the pair grid for a few confident positives

The seed (miRNA positions 2-8) is marked in the profiles. No test data is used.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import DATA_DIR, load_train
from .pairing import encode_table
from .splits import SPLITS, split_train_validation

ROOT = Path(__file__).resolve().parents[1]


def load_model(split: str, name: str, seed: int):
    path = ROOT / "artifacts" / split / f"{name}_seed{seed}.pkl"
    if not path.exists():
        print(f"skipping {name}: {path.relative_to(ROOT)} not found (train with --save)")
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def shap_m1(model, validation: pd.DataFrame, out: Path) -> None:
    import shap
    features = model.features(validation)
    values = shap.TreeExplainer(model.model).shap_values(features)
    values = values[1] if isinstance(values, list) else values
    values = values[..., 1] if values.ndim == 3 else values
    importance = pd.Series(np.abs(values).mean(0), index=features.columns).sort_values(ascending=False)
    groups = importance.groupby(importance.index.str.split("_").str[0]).sum().sort_values(ascending=False)
    importance.rename("mean_abs_shap").rename_axis("feature").to_csv(out / "m1_shap.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]})
    top = importance.head(20)[::-1]
    axes[0].barh(top.index, top.values, color="#3b6ea5")
    axes[0].set(xlabel="mean |SHAP| (log-odds)", title="M1: 20 most important features")
    axes[1].bar(groups.index, groups.values, color="#3b6ea5")
    axes[1].set(ylabel="summed mean |SHAP|", title="M1: importance per feature group")
    fig.tight_layout()
    fig.savefig(out / "m1_shap.png", dpi=150)
    plt.close(fig)
    print("M1 feature groups:", groups.round(3).to_dict())


def attention_profile(model, mirna, target, lengths) -> np.ndarray:
    """How much the target attends to each miRNA position, relative to uniform attention."""
    net = model.net.eval()
    with torch.no_grad():
        net(mirna, target)
    attention = net.target_blocks[-1].cross.last_attention.mean(1)  # batch x target x miRNA
    received = attention.mean(1) * lengths[:, None]                # 1.0 = as much as uniform attention
    return received.numpy()


def saliency(model, mirna, target) -> np.ndarray:
    """|d logit / d input grid| summed over channels (batch x miRNA x reversed target)."""
    net = model.net.eval()
    grid = net.grid(mirna, target).requires_grad_(True)
    features = net.blocks(net.stem(grid))
    pooled = torch.cat([features.amax(dim=(2, 3)), features.mean(dim=(2, 3))], dim=1)
    net.head(pooled).sum().backward()
    return grid.grad.abs().sum(1).numpy()


def masked_mean(profiles: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    mask = np.arange(profiles.shape[1])[None, :] < lengths[:, None]
    return (profiles * mask).sum(0) / np.maximum(mask.sum(0), 1)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="stratified", choices=SPLITS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args(argv)

    _, validation = split_train_validation(load_train(args.data_dir), args.split)
    out = ROOT / "results" / args.split / "interpretation"
    out.mkdir(parents=True, exist_ok=True)

    m1 = load_model(args.split, "m1_features_gbm", args.seed)
    if m1 is not None:
        shap_m1(m1, validation, out)

    positives = validation[validation["label"] == 1]
    mirna, target = map(torch.from_numpy, encode_table(positives))
    lengths = positives["mirna"].str.len().to_numpy()
    profiles = {}
    m2 = load_model(args.split, "m2_cross_attention", args.seed)
    if m2 is not None:
        profiles["M2: attention received from the target (1 = uniform)"] = masked_mean(
            attention_profile(m2, mirna, target, torch.from_numpy(lengths)), lengths)
    m3 = load_model(args.split, "m3_complementarity_cnn", args.seed)
    if m3 is not None:
        grids = saliency(m3, mirna, target)
        per_position = grids.sum(2)
        per_position = per_position / per_position.sum(1, keepdims=True) * lengths[:, None]
        profiles["M3: input-gradient saliency (1 = uniform)"] = masked_mean(per_position, lengths)

        probabilities = m3.predict_proba(positives)
        chosen = np.argsort(-probabilities)[:4]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
        for ax, row in zip(axes, chosen):
            ax.imshow(grids[row][:lengths[row]], aspect="auto", cmap="magma")
            ax.set(xlabel="target position (3'→5')", ylabel="miRNA position (5'→3')",
                   title=f"{positives['mirna'].iloc[row]}\np = {probabilities[row]:.2f}")
            ax.title.set_fontsize(8)
        fig.suptitle("M3 saliency on the pair grid: four most confident validation positives")
        fig.tight_layout()
        fig.savefig(out / "m3_saliency_examples.png", dpi=150)
        plt.close(fig)

    if profiles:
        table = pd.DataFrame(profiles, index=pd.RangeIndex(1, 29, name="mirna_position"))
        table.to_csv(out / "position_profiles.csv")
        fig, axes = plt.subplots(1, len(profiles), figsize=(7 * len(profiles), 4), squeeze=False)
        for ax, (title, values) in zip(axes[0], profiles.items()):
            ax.bar(np.arange(1, 27), values[:26], color=["#c8553d" if 2 <= p <= 8 else "#8a8f98" for p in range(1, 27)])
            ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
            ax.set(xlabel="miRNA position (seed 2-8 in red)", ylabel="relative importance", title=title)
        fig.tight_layout()
        fig.savefig(out / "position_profiles.png", dpi=150)
        plt.close(fig)
        seed_share = {title.split(":")[0]: round(float(values[1:8].mean()), 2) for title, values in profiles.items()}
        print("mean relative importance of seed positions 2-8:", seed_share)
    print(f"saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
