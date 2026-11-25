"""
Módulo de treinamento e fine-tuning da ResNet-50 na base principal (CASIA-Iris-Interval):
1. Treinamento restrito exclusivamente a protocols/train.csv (Zero Identity Leakage).
2. Suporte a repetição experimental nas 3 seeds do config.yaml ([42, 123, 2026]) com fixação completa de sementes.
3. Otimizador AdamW com taxas de aprendizado diferenciadas (lr_backbone=1e-4 para layer3/layer4;
   lr_head=3e-4 para ProjectionHead e CosFaceLoss), CosineAnnealingLR e Mixed Precision (torch.amp.autocast('cuda')).
4. Avaliação ao fim de cada época em protocols/val_pairs.csv via similaridade de cosseno para cálculo do EER de validação.
5. Salvamento de checkpoints/resnet50_casia_seed_{seed}.pth sempre que o EER de validação melhorar.
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

from model import (
    BatchHardTripletLoss,
    CosFaceLoss,
    ResNet50IrisEmbedder,
    build_model_from_config,
    get_model_info,
)
from preprocess import IrisDataset, IrisPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        path = Path(__file__).resolve().parent / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_all_seeds(seed: int) -> None:
    """Fixa todas as sementes de aleatoriedade (random, numpy, torch, cuda) para reprodutibilidade."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CachedTensorDataset(Dataset):
    """Dataset auxiliar para imagens de validação pré-processadas em memória para rápida avaliação por época."""

    def __init__(self, image_paths: List[str], preprocessor: IrisPreprocessor) -> None:
        self.image_paths = list(image_paths)
        self.tensors: List[torch.Tensor] = [
            preprocessor(p, track_fta=False) for p in self.image_paths
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        return self.tensors[idx], self.image_paths[idx]


def compute_eer_and_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Calcula o Equal Error Rate (EER, onde FMR = FNMR) e o limiar ótimo correspondente
    a partir dos rótulos binários (1 = genuine, 0 = impostor) e scores de similaridade de cosseno.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Encontra o ponto onde FMR (fpr) e FNMR (fnr) se cruzam
    diff = np.abs(fpr - fnr)
    min_idx = int(np.argmin(diff))
    eer_approx = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    best_thresh = float(thresholds[min_idx])

    try:
        eer = float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))
        thresh_interp = interp1d(fpr, thresholds, bounds_error=False, fill_value="extrapolate")
        best_thresh = float(thresh_interp(eer))
    except Exception:
        eer = eer_approx

    return eer, best_thresh


@torch.no_grad()
def evaluate_validation_eer(
    model: ResNet50IrisEmbedder,
    val_loader: DataLoader,
    val_pairs_df: pd.DataFrame,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """
    Extrai os embeddings normalizados L2 de todas as imagens de val_pairs.csv,
    calcula a similaridade de cosseno para cada par e retorna (val_eer, val_threshold).
    """
    model.eval()
    embeddings_map: Dict[str, np.ndarray] = {}

    amp_enabled = use_amp and (device.type == "cuda")
    for batch_tensors, batch_paths in val_loader:
        batch_tensors = batch_tensors.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            emb = model(batch_tensors)
        emb_np = emb.detach().cpu().numpy()
        for path_str, vec in zip(batch_paths, emb_np):
            embeddings_map[path_str] = vec

    p1_list = val_pairs_df["amostra_1"].astype(str).tolist()
    p2_list = val_pairs_df["amostra_2"].astype(str).tolist()
    tipos = val_pairs_df["tipo"].astype(str).tolist()

    emb1 = np.stack([embeddings_map[p] for p in p1_list], axis=0)
    emb2 = np.stack([embeddings_map[p] for p in p2_list], axis=0)

    # Similaridade de cosseno entre embeddings já normalizados em L2 = produto escalar
    scores = np.sum(emb1 * emb2, axis=1)
    labels = np.array([1 if t.lower() == "genuine" else 0 for t in tipos], dtype=np.int32)

    val_eer, val_thresh = compute_eer_and_threshold(labels, scores)
    return val_eer, val_thresh


def train_single_seed(
    cfg: dict,
    seed: int,
    device: torch.device,
    use_rubber_sheet: Optional[bool] = None,
    checkpoint_prefix: str = "resnet50_casia",
) -> Dict[str, float]:
    """
    Executa o ciclo completo de treinamento da ResNet-50 para uma semente específica (`seed`),
    salvando o melhor checkpoint segundo o EER de validação em `checkpoints/{checkpoint_prefix}_seed_{seed}.pth`.
    """
    set_all_seeds(seed)

    train_csv = Path("protocols/train.csv")
    val_pairs_csv = Path("protocols/val_pairs.csv")
    if not train_csv.exists() or not val_pairs_csv.exists():
        raise FileNotFoundError(
            "Protocolos 'protocols/train.csv' ou 'protocols/val_pairs.csv' não encontrados. "
            "Execute 'python generate_protocols.py' antes do treinamento."
        )

    use_rs = cfg.get("use_rubber_sheet", True) if use_rubber_sheet is None else use_rubber_sheet
    preprocessor = IrisPreprocessor.from_config(cfg, use_rubber_sheet=use_rs)

    train_dataset = IrisDataset(train_csv, preprocessor=preprocessor, is_train=True)
    num_classes = train_dataset.num_classes

    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 4))
    # Em Windows, evita overhead excessivo de multiprocessing caso dataset seja pequeno
    effective_workers = 0 if os.name == "nt" else num_workers

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=g,
    )

    val_pairs_df = pd.read_csv(val_pairs_csv)
    unique_val_paths = sorted(
        set(val_pairs_df["amostra_1"].astype(str)).union(set(val_pairs_df["amostra_2"].astype(str)))
    )
    logger.info(
        f"[Seed {seed}] Pré-carregando {len(unique_val_paths)} imagens únicas de validação (use_rubber_sheet={use_rs})..."
    )
    val_dataset = CachedTensorDataset(unique_val_paths, preprocessor=preprocessor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Instancia modelo ResNet-50 e função de perda métrica
    model = build_model_from_config(cfg, pretrained=True).to(device)
    embedding_dim = int(cfg.get("embedding_dim", 256))
    loss_type = str(cfg.get("loss_type", "cosface")).lower()
    target_margin = float(cfg.get("cosface_m", 0.35))

    if loss_type == "cosface":
        criterion: nn.Module = CosFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            s=float(cfg.get("cosface_s", 30.0)),
            m=target_margin,
        ).to(device)
    elif loss_type == "triplet":
        criterion = BatchHardTripletLoss(margin=float(cfg.get("triplet_margin", 0.3))).to(device)
    else:
        raise ValueError(f"loss_type='{loss_type}' não suportado. Use 'cosface' ou 'triplet'.")

    # Configura AdamW com taxas de aprendizado diferenciadas
    lr_backbone = float(cfg.get("lr_backbone", 1e-4))
    lr_head = float(cfg.get("lr_head", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    epochs = int(cfg.get("epochs", 25))
    use_amp = bool(cfg.get("mixed_precision", True)) and (device.type == "cuda")

    # Inicializa os protótipos de classe da CosFaceLoss com os centróides iniciais de treino (apenas train.csv)
    if loss_type == "cosface" and hasattr(criterion, "weight"):
        model.eval()
        class_sums = torch.zeros(num_classes, embedding_dim, device=device)
        class_counts = torch.zeros(num_classes, 1, device=device)
        with torch.no_grad():
            for init_imgs, init_lbls, _ in train_loader:
                init_imgs = init_imgs.to(device, non_blocking=True)
                init_lbls = init_lbls.to(device, non_blocking=True).long()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    init_emb = model(init_imgs).float()
                class_sums.index_add_(0, init_lbls, init_emb)
                class_counts.index_add_(
                    0, init_lbls, torch.ones((init_lbls.size(0), 1), device=device)
                )
        valid_mask = (class_counts.squeeze(1) > 0)
        centroids = torch.nn.functional.normalize(class_sums[valid_mask] / class_counts[valid_mask], p=2, dim=1)
        criterion.weight.data[valid_mask] = centroids

    backbone_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("projection_head")
    ]
    head_params = [p for p in model.projection_head.parameters() if p.requires_grad] + [
        p for p in criterion.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoints_dir / f"{checkpoint_prefix}_seed_{seed}.pth"

    best_val_eer = float("inf")
    best_val_thresh = 0.5
    best_epoch = 0

    logger.info(
        f"[Seed {seed}] Iniciando treinamento ResNet-50 ({epochs} épocas) | "
        f"Classes de treino: {num_classes} | Loss: {loss_type} | Device: {device}"
    )

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        criterion.train()

        running_loss = 0.0
        n_batches = 0

        for images, labels, _ in tqdm(
            train_loader,
            desc=f"[Seed {seed}] Época {epoch:02d}/{epochs:02d}",
            leave=False,
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                embeddings = model(images)
                loss = criterion(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, n_batches)

        # Avaliação no conjunto de Validação (protocols/val_pairs.csv)
        val_eer, val_thresh = evaluate_validation_eer(
            model=model,
            val_loader=val_loader,
            val_pairs_df=val_pairs_df,
            device=device,
            use_amp=use_amp,
        )
        elapsed = time.perf_counter() - t0

        improved_marker = ""
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_val_thresh = val_thresh
            best_epoch = epoch
            improved_marker = " * [MELHOR EER - SALVO]"

            ckpt_payload = {
                "seed": seed,
                "epoch": epoch,
                "best_val_eer": float(best_val_eer),
                "val_threshold": float(best_val_thresh),
                "use_rubber_sheet": bool(use_rs),
                "loss_type": loss_type,
                "model_state_dict": model.state_dict(),
                "model_info": get_model_info(model),
                "config": cfg,
            }
            torch.save(ckpt_payload, ckpt_path)

        logger.info(
            f"[Seed {seed}] Época {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
            f"Train Loss: {avg_loss:.4f} | Val EER: {val_eer * 100:.2f}% (thr={val_thresh:.4f}){improved_marker}"
        )

    logger.info(
        f"[Seed {seed}] Finalizado! Melhor Val EER: {best_val_eer * 100:.2f}% "
        f"(Época {best_epoch}, Threshold={best_val_thresh:.4f}) -> '{ckpt_path.as_posix()}'"
    )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_eer": float(best_val_eer),
        "val_threshold": float(best_val_thresh),
        "checkpoint_path": ckpt_path.as_posix(),
        "use_rubber_sheet": bool(use_rs),
    }


def run_training(
    config_path: str = "config.yaml",
    seeds_override: Optional[List[int]] = None,
    use_rubber_sheet: Optional[bool] = None,
    checkpoint_prefix: str = "resnet50_casia",
) -> Dict[str, object]:
    """
    Executa o treinamento para as 3 seeds definidas no config.yaml ([42, 123, 2026])
    e consolida média e desvio-padrão do EER de validação.
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = seeds_override if seeds_override is not None else cfg.get("seeds", [42, 123, 2026])
    seeds = [int(s) for s in seeds]

    per_seed_results = []
    for s in seeds:
        res = train_single_seed(
            cfg=cfg,
            seed=s,
            device=device,
            use_rubber_sheet=use_rubber_sheet,
            checkpoint_prefix=checkpoint_prefix,
        )
        per_seed_results.append(res)

    val_eers = [r["best_val_eer"] for r in per_seed_results]
    mean_eer = float(np.mean(val_eers))
    std_eer = float(np.std(val_eers))

    summary = {
        "checkpoint_prefix": checkpoint_prefix,
        "use_rubber_sheet": bool(
            cfg.get("use_rubber_sheet", True) if use_rubber_sheet is None else use_rubber_sheet
        ),
        "seeds": seeds,
        "val_eer_mean": round(mean_eer, 6),
        "val_eer_std": round(std_eer, 6),
        "val_eer_mean_percent": round(mean_eer * 100.0, 3),
        "val_eer_std_percent": round(std_eer * 100.0, 3),
        "runs": per_seed_results,
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_filename = (
        "training_summary.json"
        if checkpoint_prefix == "resnet50_casia"
        else f"training_summary_{checkpoint_prefix}.json"
    )
    with open(results_dir / summary_filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info(
        f"RESUMO MULTI-SEED ({seeds}) [{checkpoint_prefix}]: "
        f"Val EER = {mean_eer * 100:.2f}% ± {std_eer * 100:.2f}%"
    )
    logger.info("=" * 70)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treinamento da ResNet-50 com CosFace na base principal (CASIA-Iris-Interval)."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho para config.yaml.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Executa apenas uma seed específica. Se omitido, roda todas as seeds do config.yaml ([42, 123, 2026]).",
    )
    parser.add_argument(
        "--no-rubber-sheet",
        action="store_true",
        help="Treina no modo de Estudo de Ablação sem transformação Rubber Sheet.",
    )
    args = parser.parse_args()

    seeds_override = [args.seed] if args.seed is not None else None
    use_rs = False if args.no_rubber_sheet else None
    prefix = "resnet50_casia_no_rubbersheet" if args.no_rubber_sheet else "resnet50_casia"

    run_training(
        config_path=args.config,
        seeds_override=seeds_override,
        use_rubber_sheet=use_rs,
        checkpoint_prefix=prefix,
    )


if __name__ == "__main__":
    main()
