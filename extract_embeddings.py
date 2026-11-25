"""
Módulo de extração de embeddings normalizados em L2 e medição de eficiência computacional:
1. Carrega o checkpoint treinado da ResNet-50 (ResNet50IrisEmbedder) em modo eval().
2. Extrai os embeddings L2-normalizados para todas as imagens referenciadas em:
   - protocols/val_pairs.csv
   - protocols/test_pairs.csv (CASIA-Iris-Interval)
   - protocols/cross_dataset_test_pairs.csv (IITD - Cross-Dataset)
3. Mede o tempo médio de inferência por amostra (em milissegundos) sincronizando a GPU
   (torch.cuda.synchronize()) e identifica automaticamente o hardware (GPU e CPU).
"""

import argparse
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml

from model import ResNet50IrisEmbedder, build_model_from_config, get_model_info
from preprocess import IrisPreprocessor

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


def detect_hardware_info() -> Dict[str, str]:
    """
    Identifica automaticamente o hardware utilizado (GPU via torch.cuda.get_device_name(0) e CPU).
    """
    cpu_name = platform.processor() or platform.machine() or "Generic CPU"
    cpu_cores = os.cpu_count() or 1
    cpu_desc = f"{cpu_name} ({cpu_cores} logical cores)"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_type = "cuda"
        summary = f"GPU: {gpu_name} | CPU: {cpu_desc}"
    else:
        gpu_name = "N/A (CPU only)"
        device_type = "cpu"
        summary = f"CPU: {cpu_desc}"

    return {
        "device_type": device_type,
        "gpu_name": gpu_name,
        "cpu_name": cpu_desc,
        "hardware_summary": summary,
        "pytorch_version": torch.__version__,
    }


def load_trained_model(
    cfg: dict,
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> Tuple[ResNet50IrisEmbedder, dict]:
    """
    Carrega a ResNet50IrisEmbedder e os pesos do checkpoint em modo eval().
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Checkpoint '{ckpt_file}' não encontrado. Execute 'python train.py' primeiro."
        )

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    model = build_model_from_config(cfg, pretrained=False).to(device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, ckpt


def collect_unique_protocol_images(
    protocols_dir: Union[str, Path] = "protocols"
) -> List[str]:
    """
    Coleta todas as imagens únicas referenciadas em val_pairs.csv, test_pairs.csv e cross_dataset_test_pairs.csv.
    """
    p_dir = Path(protocols_dir)
    pair_files = [
        p_dir / "val_pairs.csv",
        p_dir / "test_pairs.csv",
        p_dir / "cross_dataset_test_pairs.csv",
    ]

    unique_paths: Set[str] = set()
    for pf in pair_files:
        if pf.exists():
            df = pd.read_csv(pf)
            unique_paths.update(df["amostra_1"].astype(str).tolist())
            unique_paths.update(df["amostra_2"].astype(str).tolist())

    return sorted(unique_paths)


@torch.no_grad()
def extract_embeddings_for_paths(
    model: ResNet50IrisEmbedder,
    image_paths: List[str],
    preprocessor: IrisPreprocessor,
    device: torch.device,
    batch_size: int = 64,
    use_amp: bool = True,
    desc: str = "Extraindo embeddings",
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Extrai os embeddings L2-normalizados para uma lista de caminhos de imagem e mede
    o tempo médio de inferência por amostra (em milissegundos) sincronizando a GPU.
    """
    model.eval()
    amp_enabled = use_amp and (device.type == "cuda")

    # Warm-up da GPU/CPU para estabilizar medição de latência
    dummy = torch.zeros((min(8, max(1, len(image_paths))), 3, 224, 224), device=device)
    for _ in range(3):
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    embeddings_dict: Dict[str, np.ndarray] = {}
    total_cnn_time_sec = 0.0
    total_pipeline_time_sec = 0.0
    total_samples = len(image_paths)

    for start_idx in tqdm(range(0, total_samples, batch_size), desc=desc, leave=False):
        batch_paths = image_paths[start_idx : start_idx + batch_size]

        t_prep_start = time.perf_counter()
        batch_tensors_list = [preprocessor(p, track_fta=True) for p in batch_paths]
        batch_tensor = torch.stack(batch_tensors_list, dim=0).to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_cnn_start = time.perf_counter()

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            batch_emb = model(batch_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_cnn_end = time.perf_counter()

        total_cnn_time_sec += t_cnn_end - t_cnn_start
        total_pipeline_time_sec += t_cnn_end - t_prep_start

        emb_np = batch_emb.detach().cpu().numpy().astype(np.float32)
        for p, vec in zip(batch_paths, emb_np):
            embeddings_dict[p] = vec

    avg_cnn_ms = (total_cnn_time_sec / max(1, total_samples)) * 1000.0
    avg_total_ms = (total_pipeline_time_sec / max(1, total_samples)) * 1000.0

    timing_stats = {
        "total_samples": int(total_samples),
        "avg_cnn_inference_ms": round(float(avg_cnn_ms), 4),
        "avg_total_pipeline_ms": round(float(avg_total_ms), 4),
    }
    return embeddings_dict, timing_stats


def run_extraction(
    config_path: str = "config.yaml",
    seed: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    use_rubber_sheet: Optional[bool] = None,
    output_npz: Optional[str] = None,
) -> Dict[str, object]:
    """
    Executa a extração completa de embeddings para val_pairs.csv, test_pairs.csv e cross_dataset_test_pairs.csv,
    salvando os vetores em arquivo .npz e os metadados de eficiência e FTA em results/.
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware_info = detect_hardware_info()

    target_seed = int(seed if seed is not None else cfg.get("protocol_seed", 42))
    if checkpoint_path is None:
        prefix = "resnet50_casia_no_rubbersheet" if use_rubber_sheet is False else "resnet50_casia"
        ckpt_path = Path("checkpoints") / f"{prefix}_seed_{target_seed}.pth"
    else:
        ckpt_path = Path(checkpoint_path)

    model, ckpt = load_trained_model(cfg, ckpt_path, device=device)
    use_rs = (
        bool(ckpt.get("use_rubber_sheet", cfg.get("use_rubber_sheet", True)))
        if use_rubber_sheet is None
        else bool(use_rubber_sheet)
    )

    preprocessor = IrisPreprocessor.from_config(cfg, use_rubber_sheet=use_rs)
    unique_paths = collect_unique_protocol_images("protocols")

    if not unique_paths:
        raise RuntimeError(
            "Nenhuma imagem encontrada nos protocolos em 'protocols/'. Execute 'python generate_protocols.py'."
        )

    logger.info(
        f"Extraindo embeddings de {len(unique_paths)} imagens únicas | "
        f"Checkpoint: '{ckpt_path.as_posix()}' | Hardware: {hardware_info['hardware_summary']}"
    )

    embeddings_dict, timing_stats = extract_embeddings_for_paths(
        model=model,
        image_paths=unique_paths,
        preprocessor=preprocessor,
        device=device,
        batch_size=int(cfg.get("batch_size", 64)),
        use_amp=bool(cfg.get("mixed_precision", True)),
    )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if output_npz is None:
        suffix = f"seed_{target_seed}" if use_rs else f"no_rubbersheet_seed_{target_seed}"
        npz_path = results_dir / f"embeddings_{suffix}.npz"
    else:
        npz_path = Path(output_npz)

    np.savez_compressed(npz_path, **embeddings_dict)

    # Se for a semente padrão (42) com Rubber Sheet, salva também como embeddings.npz padrão
    if target_seed == int(cfg.get("protocol_seed", 42)) and use_rs:
        np.savez_compressed(results_dir / "embeddings.npz", **embeddings_dict)

    # Salva relatórios de FTA e metadados de eficiência
    preprocessor.fta_tracker.save_json(results_dir / "fta_report.json")
    model_info = get_model_info(model, checkpoint_path=ckpt_path)

    extraction_metadata = {
        "seed": target_seed,
        "checkpoint_path": ckpt_path.as_posix(),
        "use_rubber_sheet": use_rs,
        "val_threshold_from_ckpt": float(ckpt.get("val_threshold", 0.5)),
        "best_val_eer_from_ckpt": float(ckpt.get("best_val_eer", 0.0)),
        "model_info": model_info,
        "timing": timing_stats,
        "hardware": hardware_info,
        "fta": preprocessor.fta_tracker.to_dict(),
    }

    meta_filename = (
        f"extraction_metadata_seed_{target_seed}.json"
        if use_rs
        else f"extraction_metadata_no_rubbersheet_seed_{target_seed}.json"
    )
    with open(results_dir / meta_filename, "w", encoding="utf-8") as f:
        json.dump(extraction_metadata, f, indent=2, ensure_ascii=False)

    if target_seed == int(cfg.get("protocol_seed", 42)) and use_rs:
        with open(results_dir / "extraction_metadata.json", "w", encoding="utf-8") as f:
            json.dump(extraction_metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Extração concluída! Tempo médio CNN: {timing_stats['avg_cnn_inference_ms']:.3f} ms/amostra | "
        f"Pipeline total: {timing_stats['avg_total_pipeline_ms']:.3f} ms/amostra | "
        f"FTA: {preprocessor.fta_tracker.fta_count}/{preprocessor.fta_tracker.total_evaluated} "
        f"({preprocessor.fta_tracker.fta_rate * 100:.2f}%)"
    )

    return {
        "embeddings": embeddings_dict,
        "metadata": extraction_metadata,
        "npz_path": npz_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai embeddings L2-normalizados da ResNet-50 e mede tempo médio de inferência."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho para config.yaml.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed específica a extrair. Se omitido e --all-seeds ativo, extrai todas as seeds do config.yaml.",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Extrai embeddings para todas as seeds listadas no config.yaml ([42, 123, 2026]).",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Caminho manual para um checkpoint .pth.")
    parser.add_argument(
        "--no-rubber-sheet",
        action="store_true",
        help="Usa modo de Estudo de Ablação sem Rubber Sheet.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_rs = False if args.no_rubber_sheet else None

    if args.all_seeds:
        for s in cfg.get("seeds", [42, 123, 2026]):
            run_extraction(
                config_path=args.config,
                seed=int(s),
                use_rubber_sheet=use_rs,
            )
    else:
        run_extraction(
            config_path=args.config,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            use_rubber_sheet=use_rs,
        )


if __name__ == "__main__":
    main()
