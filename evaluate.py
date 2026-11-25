"""
Módulo de avaliação biométrica 1:1, Experimento A (Base Principal CASIA), Experimento C (Cross-Dataset IITD)
e Análise Automática de Erros (3 Falsos Matches e 3 Falsos Non-Matches):
1. Matching via Similaridade de Cosseno entre embeddings L2-normalizados: s = dot(emb_A, emb_B).
2. Definição do threshold (tau*) exclusivamente na Validação da CASIA (ponto onde FMR = FNMR) e CONGELAMENTO.
3. Avaliação multi-seed (42, 123, 2026) no Teste da CASIA (Experimento A), salvando results/scores.csv,
   results/metrics.json (Tabela Padronizada completa), results/roc.png e results/distributions.png.
4. Avaliação Cross-Dataset (Experimento C: CASIA -> IITD sem fine-tuning) em results/cross_dataset_iitd/.
5. Seleção e diagnóstico automático de 3 Falsos Matches e 3 Falsos Non-Matches com indicadores físicos
   (foco Laplaciano, reflexo especular, razão de dilatação rp/ri e oclusão de pálpebra/cílios)
   em results/error_analysis.csv e prancha visual em results/error_analysis.png.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, roc_curve
import torch
import yaml

from extract_embeddings import (
    detect_hardware_info,
    extract_embeddings_for_paths,
    load_trained_model,
    run_extraction,
)
from model import build_model_from_config, get_model_info
from preprocess import (
    IrisPreprocessor,
    SegmentationResult,
    draw_segmentation_overlay,
    segment_iris,
)

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


def compute_pair_scores(
    pairs_df: pd.DataFrame, embeddings_map: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Calcula a similaridade de cosseno s = dot(emb_A, emb_B) para todos os pares de pairs_df.
    Retorna (labels_binarios, scores_float, df_com_scores).
    """
    p1_list = pairs_df["amostra_1"].astype(str).tolist()
    p2_list = pairs_df["amostra_2"].astype(str).tolist()
    tipos = pairs_df["tipo"].astype(str).tolist()

    emb1 = np.stack([embeddings_map[p] for p in p1_list], axis=0)
    emb2 = np.stack([embeddings_map[p] for p in p2_list], axis=0)

    # Garante normalização L2 caso algum vetor venha de arquivo externo
    emb1 = emb1 / np.clip(np.linalg.norm(emb1, axis=1, keepdims=True), 1e-12, None)
    emb2 = emb2 / np.clip(np.linalg.norm(emb2, axis=1, keepdims=True), 1e-12, None)

    scores = np.sum(emb1 * emb2, axis=1).astype(np.float64)
    labels = np.array([1 if t.lower() == "genuine" else 0 for t in tipos], dtype=np.int32)

    scored_df = pairs_df.copy()
    scored_df["score"] = scores
    return labels, scores, scored_df


def find_validation_threshold_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    2. Definição do Threshold (tau*) na Validação:
       Encontra o limiar tau* no ponto onde FMR = FNMR (EER de validação).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    diff = np.abs(fpr - fnr)
    min_idx = int(np.argmin(diff))
    eer_val = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    tau_star = float(thresholds[min_idx])

    try:
        eer_val = float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))
        thresh_interp = interp1d(fpr, thresholds, bounds_error=False, fill_value="extrapolate")
        tau_star = float(thresh_interp(eer_val))
    except Exception:
        pass

    return tau_star, eer_val


def compute_biometric_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    frozen_threshold: float,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calcula todas as métricas biométricas obrigatórias:
    - FMR (FAR) e FNMR (FRR) operacionais aplicando o threshold congelado tau* da validação
    - EER do conjunto avaliado
    - Curva ROC (fpr, tpr, thresholds) e AUC
    - TAR@FAR=1% (True Acceptance Rate quando FAR = 1%)
    """
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    # Taxas operacionais com o threshold congelado tau* da validação
    fmr_operational = (
        float(np.mean(impostor_scores >= frozen_threshold)) if len(impostor_scores) > 0 else 0.0
    )
    fnmr_operational = (
        float(np.mean(genuine_scores < frozen_threshold)) if len(genuine_scores) > 0 else 0.0
    )

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    roc_auc = float(auc(fpr, tpr))

    # EER (ponto onde FMR == FNMR)
    diff = np.abs(fpr - fnr)
    min_idx = int(np.argmin(diff))
    eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    eer_threshold = float(thresholds[min_idx])
    try:
        eer = float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))
    except Exception:
        pass

    # TAR@FAR=1% (FAR = 0.01)
    try:
        tar_at_far_1pct = float(interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))(0.01))
    except Exception:
        valid_idx = np.where(fpr <= 0.01)[0]
        tar_at_far_1pct = float(tpr[valid_idx[-1]]) if len(valid_idx) > 0 else 0.0

    return {
        "threshold_congelado": float(frozen_threshold),
        "FMR": float(fmr_operational),
        "FNMR": float(fnmr_operational),
        "EER": float(eer),
        "eer_threshold": float(eer_threshold),
        "AUC": float(roc_auc),
        "TAR_at_FAR_1pct": float(tar_at_far_1pct),
        "fpr": fpr,
        "tpr": tpr,
        "fnr": fnr,
    }


def format_sample_name_for_csv(path_str: str, is_casia: bool = True) -> str:
    """
    Formata o nome da amostra para scores.csv:
    - Na CASIA-Iris-Interval: S1001L01.jpg
    - Na IITD: Left/001_1.bmp ou 001/01_L.bmp (preserva subpasta para distinguir Left/Right)
    """
    p = Path(path_str)
    if is_casia:
        return p.name
    if p.parent.name and (p.parent.name.isdigit() or p.parent.name.lower() in {"left", "right", "l", "r"}):
        return f"{p.parent.name}/{p.name}"
    return p.name


def save_scores_csv(
    scored_df: pd.DataFrame, output_csv: Path, is_casia: bool = True
) -> None:
    """
    Salva scores.csv exatamente no formato:
    amostra_1,amostra_2,tipo,score
    S1001L01.jpg,S1001L02.jpg,genuine,0.8721
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        {
            "amostra_1": [format_sample_name_for_csv(p, is_casia=is_casia) for p in scored_df["amostra_1"]],
            "amostra_2": [format_sample_name_for_csv(p, is_casia=is_casia) for p in scored_df["amostra_2"]],
            "tipo": scored_df["tipo"].astype(str),
            "score": [f"{float(s):.4f}" for s in scored_df["score"]],
        }
    )
    out_df.to_csv(output_csv, index=False)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    eer: float,
    title: str,
    output_path: Path,
) -> None:
    """Gera o gráfico results/roc.png com curva ROC, AUC, linha de EER e ponto de EER destacados."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=220)

    ax.plot(
        fpr * 100.0,
        tpr * 100.0,
        color="#1565C0",
        lw=2.4,
        label=f"Curva ROC (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 100], [0, 100], color="#9E9E9E", lw=1.2, linestyle="--", label="Aleatório (AUC = 0.5000)")

    # Marca o ponto de EER (FMR = FNMR => TPR = 1 - EER)
    eer_x = eer * 100.0
    eer_y = (1.0 - eer) * 100.0
    ax.scatter(
        [eer_x],
        [eer_y],
        color="#D32F2F",
        s=75,
        zorder=5,
        label=f"EER = {eer * 100.0:.2f}%",
    )
    ax.axvline(x=1.0, color="#2E7D32", linestyle=":", lw=1.3, label="FAR = 1.0%")

    ax.set_xlim([-1.0, 100.0])
    ax.set_ylim([0.0, 101.0])
    ax.set_xlabel("Taxa de Falsa Aceitação — FMR / FAR (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Taxa de Reconhecimento Genuíno — TAR / 1 - FNMR (%)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(
    labels: np.ndarray,
    scores: np.ndarray,
    frozen_threshold: float,
    eer: float,
    title: str,
    output_path: Path,
) -> None:
    """
    Gera results/distributions.png com histograma/densidade de scores Genuínos vs Impostores
    e linha vertical tracejada indicando o threshold congelado tau*.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=220)

    sns.histplot(
        impostor_scores,
        bins=50,
        stat="density",
        kde=True,
        color="#E53935",
        alpha=0.40,
        label=f"Impostor (n={len(impostor_scores)})",
        ax=ax,
    )
    sns.histplot(
        genuine_scores,
        bins=50,
        stat="density",
        kde=True,
        color="#1E88E5",
        alpha=0.40,
        label=f"Genuíno (n={len(genuine_scores)})",
        ax=ax,
    )

    ax.axvline(
        x=frozen_threshold,
        color="#212121",
        linestyle="--",
        linewidth=2.2,
        label=f"Threshold Congelado (τ* = {frozen_threshold:.4f} | EER = {eer * 100.0:.2f}%)",
    )

    ax.set_xlabel("Score de Similaridade de Cosseno", fontsize=11, fontweight="bold")
    ax.set_ylabel("Densidade", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _compute_image_diagnostic_indicators(img_path: str) -> Dict[str, Union[float, SegmentationResult, np.ndarray]]:
    """
    Calcula indicadores físicos de uma amostra de íris para fundamentar a Análise de Erros:
    - Nitidez (variância do Laplaciano)
    - Percentual de pixels saturados / reflexo especular (> 240)
    - Razão de dilatação pupilar (rp / ri)
    - Percentual estimado de oclusão por pálpebra/cílios nos setores superior e inferior do anel da íris
    """
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        gray = np.zeros((224, 224), dtype=np.uint8)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    specular_pct = float(np.mean(gray > 240) * 100.0)

    _, seg = segment_iris(gray)
    dilation_ratio = float(seg.rp / max(1.0, seg.ri))

    # Estima oclusão de pálpebras/cílios nos arcos superior e inferior do anel da íris
    h, w = gray.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dist_i = np.sqrt((xx - seg.xi) ** 2 + (yy - seg.yi) ** 2)
    dist_p = np.sqrt((xx - seg.xp) ** 2 + (yy - seg.yp) ** 2)
    annulus_mask = (dist_i <= seg.ri) & (dist_p >= seg.rp)

    # Regiões verticalmente extremas do anel (30% superior e 30% inferior) onde pálpebras e cílios ocorrem
    vertical_offset = np.abs(yy - seg.yi)
    top_bottom_sector = annulus_mask & (vertical_offset >= 0.55 * seg.ri)
    mid_lateral_sector = annulus_mask & (vertical_offset < 0.40 * seg.ri)

    if np.count_nonzero(top_bottom_sector) > 10 and np.count_nonzero(mid_lateral_sector) > 10:
        lateral_vals = gray[mid_lateral_sector]
        p10, p90 = np.percentile(lateral_vals, [10, 90])
        tb_vals = gray[top_bottom_sector]
        # Pixels muito mais escuros (cílios/sombra) ou muito mais claros (pálpebra/pele refletida) que o estroma lateral
        occluded_pixels = (tb_vals < max(35.0, p10 - 20.0)) | (tb_vals > min(230.0, p90 + 25.0))
        eyelid_occlusion_pct = float(np.mean(occluded_pixels) * 100.0)
    else:
        eyelid_occlusion_pct = 0.0

    return {
        "gray": gray,
        "seg": seg,
        "laplacian_var": round(lap_var, 2),
        "specular_pct": round(specular_pct, 3),
        "dilation_ratio": round(dilation_ratio, 4),
        "eyelid_occlusion_pct": round(eyelid_occlusion_pct, 2),
    }


def _synthesize_error_hypothesis(
    error_type: str,
    ind1: dict,
    ind2: dict,
    score: float,
    threshold: float,
) -> str:
    """
    Gera uma hipótese técnica fundamentada nos indicadores físicos medidos no par.
    """
    dilation_diff = abs(ind1["dilation_ratio"] - ind2["dilation_ratio"])
    max_occlusion = max(ind1["eyelid_occlusion_pct"], ind2["eyelid_occlusion_pct"])
    min_sharpness = min(ind1["laplacian_var"], ind2["laplacian_var"])
    max_specular = max(ind1["specular_pct"], ind2["specular_pct"])
    used_fb = ind1["seg"].used_limbus_fallback or ind2["seg"].used_limbus_fallback

    reasons = []
    if error_type == "False Non-Match (FNMR)":
        if dilation_diff >= 0.08:
            reasons.append(
                f"Variação acentuada de dilatação pupilar (Δrp/ri={dilation_diff:.2f}) deformando não-linearmente o estroma"
            )
        if max_occlusion >= 18.0:
            reasons.append(
                f"Oclusão severa por pálpebras/cílios ({max_occlusion:.1f}%) mascarando faixas superiores/inferiores no Rubber Sheet"
            )
        if min_sharpness < 25.0:
            reasons.append(
                f"Desfocagem óptica / baixo contraste da textura da íris (Laplaciano={min_sharpness:.1f})"
            )
        if used_fb:
            reasons.append("Imprecisão na localização da borda do limbo (acionamento de fallback geométrico)")
        if max_specular > 1.2:
            reasons.append(f"Reflexos especulares intensos ({max_specular:.2f}% saturado)")
        if not reasons:
            reasons.append(
                f"Rotação ocular torsional residual e variação de iluminação NIR entre capturas (Δrp/ri={dilation_diff:.2f}, oclusão={max_occlusion:.1f}%)"
            )
    else:
        # False Match (FMR)
        if max_occlusion >= 20.0:
            reasons.append(
                f"Artefatos comuns de oclusão palpebral/cílios ({max_occlusion:.1f}%) gerando padrões espúrios correlacionados na faixa polar"
            )
        if min_sharpness < 30.0:
            reasons.append(
                f"Suavização de alta frequência (Laplaciano={min_sharpness:.1f}) reduzindo detalhes finos discriminativos entre indivíduos"
            )
        if dilation_diff < 0.04:
            reasons.append(
                f"Geometria pupilar/limbo quase idêntica (Δrp/ri={dilation_diff:.2f}) e macro-textura radial semelhante após CLAHE"
            )
        if not reasons:
            reasons.append(
                f"Similaridade estrutural de baixa frequência no estroma e iluminação NIR homogênea próxima ao limiar (score={score:.4f} vs τ*={threshold:.4f})"
            )

    return "; ".join(reasons[:2]) + "."


def run_error_analysis(
    scored_test_df: pd.DataFrame,
    frozen_threshold: float,
    output_csv: Path = Path("results/error_analysis.csv"),
    output_png: Path = Path("results/error_analysis.png"),
) -> pd.DataFrame:
    """
    5. Análise Automática dos Erros:
       Seleciona pelo menos 3 Falsos Matches (impostor com score >= threshold_congelado)
       e 3 Falsos Non-Matches (genuine com score < threshold_congelado) no teste da CASIA,
       calcula indicadores técnicos e gera results/error_analysis.csv e results/error_analysis.png.
    """
    impostors = scored_test_df[scored_test_df["tipo"] == "impostor"].sort_values(
        by="score", ascending=False
    )
    genuines = scored_test_df[scored_test_df["tipo"] == "genuine"].sort_values(
        by="score", ascending=True
    )

    # Seleciona os 3 Falsos Matches mais críticos (ou os 3 impostores de maior score caso haja < 3 acima de tau*)
    fm_above = impostors[impostors["score"] >= frozen_threshold]
    false_matches = fm_above.head(3) if len(fm_above) >= 3 else impostors.head(3)

    # Seleciona os 3 Falsos Non-Matches mais críticos (ou os 3 genuínos de menor score caso haja < 3 abaixo de tau*)
    fnm_below = genuines[genuines["score"] < frozen_threshold]
    false_non_matches = fnm_below.head(3) if len(fnm_below) >= 3 else genuines.head(3)

    cases = []
    for idx, (_, row) in enumerate(false_matches.iterrows(), start=1):
        cases.append(("False Match (FMR)", f"FM #{idx}", row))
    for idx, (_, row) in enumerate(false_non_matches.iterrows(), start=1):
        cases.append(("False Non-Match (FNMR)", f"FNM #{idx}", row))

    records = []
    visual_items = []

    for err_type, short_code, row in cases:
        p1 = str(row["amostra_1"])
        p2 = str(row["amostra_2"])
        score = float(row["score"])

        ind1 = _compute_image_diagnostic_indicators(p1)
        ind2 = _compute_image_diagnostic_indicators(p2)
        hypothesis = _synthesize_error_hypothesis(err_type, ind1, ind2, score, frozen_threshold)

        records.append(
            {
                "caso": short_code,
                "tipo_erro": err_type,
                "amostra_1": Path(p1).name,
                "amostra_2": Path(p2).name,
                "tipo_par_real": str(row["tipo"]),
                "score": round(score, 4),
                "threshold_congelado": round(float(frozen_threshold), 4),
                "laplacian_var_1": ind1["laplacian_var"],
                "laplacian_var_2": ind2["laplacian_var"],
                "specular_pct_1": ind1["specular_pct"],
                "specular_pct_2": ind2["specular_pct"],
                "rp_ri_ratio_1": ind1["dilation_ratio"],
                "rp_ri_ratio_2": ind2["dilation_ratio"],
                "dilation_diff": round(abs(ind1["dilation_ratio"] - ind2["dilation_ratio"]), 4),
                "eyelid_occlusion_pct_1": ind1["eyelid_occlusion_pct"],
                "eyelid_occlusion_pct_2": ind2["eyelid_occlusion_pct"],
                "hipotese_tecnica": hypothesis,
            }
        )
        visual_items.append((short_code, err_type, p1, p2, score, ind1, ind2, hypothesis))

    df_errors = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_errors.to_csv(output_csv, index=False)

    # Prancha visual results/error_analysis.png (6 linhas x 2 colunas com segmentação e diagnóstico)
    n_cases = len(visual_items)
    if n_cases > 0:
        fig, axes = plt.subplots(n_cases, 2, figsize=(13.5, 2.85 * n_cases), dpi=200)
        if n_cases == 1:
            axes = np.expand_dims(axes, axis=0)

        for r_idx, (short_code, err_type, p1, p2, score, ind1, ind2, hypothesis) in enumerate(visual_items):
            ov1 = draw_segmentation_overlay(ind1["gray"], ind1["seg"])
            ov2 = draw_segmentation_overlay(ind2["gray"], ind2["seg"])

            ax_left = axes[r_idx, 0]
            ax_right = axes[r_idx, 1]

            ax_left.imshow(ov1)
            ax_left.set_title(
                f"[{short_code} - {err_type}] Amostra 1: {Path(p1).name}\n"
                f"Foco={ind1['laplacian_var']:.1f} | rp/ri={ind1['dilation_ratio']:.2f} | Oclusão={ind1['eyelid_occlusion_pct']:.1f}%",
                fontsize=9,
                fontweight="bold",
            )
            ax_left.set_xticks([])
            ax_left.set_yticks([])

            ax_right.imshow(ov2)
            ax_right.set_title(
                f"Amostra 2: {Path(p2).name}  |  Score={score:.4f} (τ*={frozen_threshold:.4f})\n"
                f"Foco={ind2['laplacian_var']:.1f} | rp/ri={ind2['dilation_ratio']:.2f} | Oclusão={ind2['eyelid_occlusion_pct']:.1f}%",
                fontsize=9,
                fontweight="bold",
            )
            ax_right.set_xlabel(f"Hipótese: {hypothesis}", fontsize=8.5, style="italic")
            ax_right.set_xticks([])
            ax_right.set_yticks([])

        plt.suptitle(
            f"Análise Crítica de Erros — 3 Falsos Matches (FM) e 3 Falsos Non-Matches (FNM) | τ* = {frozen_threshold:.4f}",
            fontsize=12.5,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, bbox_inches="tight")
        plt.close(fig)

    logger.info(
        f"Análise de erros salva em '{output_csv.as_posix()}' e '{output_png.as_posix()}' ({len(df_errors)} casos)."
    )
    return df_errors


def _load_or_extract_seed_embeddings(
    cfg: dict, seed: int, use_rubber_sheet: bool = True
) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Carrega os embeddings já extraídos em results/ ou executa a extração automaticamente caso não existam.
    """
    results_dir = Path("results")
    suffix = f"seed_{seed}" if use_rubber_sheet else f"no_rubbersheet_seed_{seed}"
    npz_path = results_dir / f"embeddings_{suffix}.npz"
    meta_path = (
        results_dir / f"extraction_metadata_seed_{seed}.json"
        if use_rubber_sheet
        else results_dir / f"extraction_metadata_no_rubbersheet_seed_{seed}.json"
    )

    if npz_path.exists() and meta_path.exists():
        data = np.load(npz_path)
        embeddings_map = {k: data[k] for k in data.files}
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return embeddings_map, metadata

    res = run_extraction(
        seed=seed,
        use_rubber_sheet=use_rubber_sheet,
    )
    return res["embeddings"], res["metadata"]


def run_evaluation(
    config_path: str = "config.yaml",
    use_rubber_sheet: Optional[bool] = None,
    results_subdir: Optional[str] = None,
) -> Dict[str, object]:
    """
    Executa a avaliação completa:
    - Experimento A: Avaliação multi-seed (42, 123, 2026) na base principal CASIA-Iris-Interval
      com threshold congelado na validação.
    - Experimento C: Avaliação Cross-Dataset (CASIA -> IITD sem fine-tuning).
    - Análise de Erros (3 Falsos Matches + 3 Falsos Non-Matches).
    """
    cfg = load_config(config_path)
    use_rs = bool(cfg.get("use_rubber_sheet", True) if use_rubber_sheet is None else use_rubber_sheet)
    out_dir = Path(results_subdir) if results_subdir else Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    val_pairs_df = pd.read_csv("protocols/val_pairs.csv")
    test_pairs_df = pd.read_csv("protocols/test_pairs.csv")
    cross_pairs_path = Path("protocols/cross_dataset_test_pairs.csv")
    has_cross = cross_pairs_path.exists()
    cross_pairs_df = pd.read_csv(cross_pairs_path) if has_cross else None

    configured_seeds = [int(s) for s in cfg.get("seeds", [42, 123, 2026])]
    prefix = "resnet50_casia" if use_rs else "resnet50_casia_no_rubbersheet"
    available_seeds = [
        s for s in configured_seeds if (Path("checkpoints") / f"{prefix}_seed_{s}.pth").exists()
    ]
    if not available_seeds:
        available_seeds = [int(cfg.get("protocol_seed", 42))]

    casia_seed_metrics = []
    iitd_seed_metrics = []

    primary_seed = available_seeds[0]
    primary_test_scored_df = None
    primary_test_labels = None
    primary_test_scores = None
    primary_test_eval = None
    primary_meta = None

    primary_cross_scored_df = None
    primary_cross_labels = None
    primary_cross_scores = None
    primary_cross_eval = None

    for s in available_seeds:
        emb_map, meta = _load_or_extract_seed_embeddings(cfg, seed=s, use_rubber_sheet=use_rs)

        # 1. Define e CONGELA o threshold tau* exclusivamente nos pares de Validação da CASIA
        val_labels, val_scores, _ = compute_pair_scores(val_pairs_df, emb_map)
        tau_star, val_eer = find_validation_threshold_eer(val_labels, val_scores)

        # 2. Avalia no conjunto de Teste da CASIA (Experimento A) com tau* congelado
        test_labels, test_scores, test_scored_df = compute_pair_scores(test_pairs_df, emb_map)
        test_eval = compute_biometric_metrics(test_labels, test_scores, frozen_threshold=tau_star)

        casia_seed_metrics.append(
            {
                "seed": s,
                "val_eer": round(val_eer, 6),
                "threshold_congelado": round(tau_star, 6),
                "EER": round(test_eval["EER"], 6),
                "AUC": round(test_eval["AUC"], 6),
                "FMR": round(test_eval["FMR"], 6),
                "FNMR": round(test_eval["FNMR"], 6),
                "TAR_at_FAR_1pct": round(test_eval["TAR_at_FAR_1pct"], 6),
                "tempo_inferencia_ms": meta["timing"]["avg_cnn_inference_ms"],
            }
        )

        if s == primary_seed:
            primary_test_scored_df = test_scored_df
            primary_test_labels = test_labels
            primary_test_scores = test_scores
            primary_test_eval = test_eval
            primary_meta = meta

        # 3. Experimento C: Cross-Dataset na IITD sem fine-tuning
        if has_cross and cross_pairs_df is not None:
            c_labels, c_scores, c_scored_df = compute_pair_scores(cross_pairs_df, emb_map)
            c_eval = compute_biometric_metrics(c_labels, c_scores, frozen_threshold=tau_star)
            iitd_seed_metrics.append(
                {
                    "seed": s,
                    "threshold_congelado_casia": round(tau_star, 6),
                    "EER": round(c_eval["EER"], 6),
                    "AUC": round(c_eval["AUC"], 6),
                    "FMR_with_casia_threshold": round(c_eval["FMR"], 6),
                    "FNMR_with_casia_threshold": round(c_eval["FNMR"], 6),
                    "TAR_at_FAR_1pct": round(c_eval["TAR_at_FAR_1pct"], 6),
                }
            )
            if s == primary_seed:
                primary_cross_scored_df = c_scored_df
                primary_cross_labels = c_labels
                primary_cross_scores = c_scores
                primary_cross_eval = c_eval

    # Consolidação de média e desvio-padrão das seeds no Teste da CASIA (Experimento A)
    eers = [m["EER"] for m in casia_seed_metrics]
    aucs = [m["AUC"] for m in casia_seed_metrics]
    fmrs = [m["FMR"] for m in casia_seed_metrics]
    fnmrs = [m["FNMR"] for m in casia_seed_metrics]
    tars = [m["TAR_at_FAR_1pct"] for m in casia_seed_metrics]
    taus = [m["threshold_congelado"] for m in casia_seed_metrics]
    inf_times = [m["tempo_inferencia_ms"] for m in casia_seed_metrics]

    model_info = primary_meta["model_info"]
    hw_info = primary_meta["hardware"]
    fta_info = primary_meta["fta"]

    loss_name = (
        "CosFace (s=30.0, m=0.35)"
        if str(cfg.get("loss_type", "cosface")).lower() == "cosface"
        else "BatchHardTripletLoss (Cosine, m=0.30)"
    )

    metrics_payload = {
        "modalidade": "Íris",
        "dataset_principal": "CASIA-Iris-V3 Interval",
        "dataset_secundario": "IIT Delhi Iris",
        "cnn": "ResNet-50",
        "pre_treinamento": "ImageNet-1K (IMAGENET1K_V2)",
        "camadas_congeladas": model_info["frozen_layers"],
        "camadas_treinadas": model_info["trained_layers"],
        "funcao_perda": loss_name,
        "num_parametros_total": model_info["total_params"],
        "num_parametros_treinaveis": model_info["trainable_params"],
        "dimensao_embedding": model_info["embedding_dim"],
        "matching": "Cosine Similarity",
        "use_rubber_sheet": use_rs,
        "seeds_avaliadas": available_seeds,
        "threshold_congelado": round(float(primary_test_eval["threshold_congelado"]), 4),
        "threshold_congelado_mean": round(float(np.mean(taus)), 4),
        "EER_mean": round(float(np.mean(eers)), 4),
        "EER_std": round(float(np.std(eers)), 4),
        "AUC_mean": round(float(np.mean(aucs)), 4),
        "AUC_std": round(float(np.std(aucs)), 4),
        "FMR_mean": round(float(np.mean(fmrs)), 4),
        "FMR_std": round(float(np.std(fmrs)), 4),
        "FNMR_mean": round(float(np.mean(fnmrs)), 4),
        "FNMR_std": round(float(np.std(fnmrs)), 4),
        "TAR_at_FAR_1pct_mean": round(float(np.mean(tars)), 4),
        "TAR_at_FAR_1pct_std": round(float(np.std(tars)), 4),
        "FTA_rate": round(float(fta_info["fta_rate"]), 6),
        "tempo_inferencia_ms": round(float(np.mean(inf_times)), 4),
        "tamanho_modelo_mb": model_info["model_size_mb"],
        "hardware": hw_info["hardware_summary"],
        "per_seed_results": casia_seed_metrics,
    }

    # Salva artefatos principais do Experimento A em out_dir (por padrão results/)
    save_scores_csv(primary_test_scored_df, out_dir / "scores.csv", is_casia=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    plot_roc_curve(
        fpr=primary_test_eval["fpr"],
        tpr=primary_test_eval["tpr"],
        roc_auc=primary_test_eval["AUC"],
        eer=primary_test_eval["EER"],
        title="Curva ROC — ResNet-50 em CASIA-Iris-Interval (Teste)",
        output_path=out_dir / "roc.png",
    )
    plot_score_distributions(
        labels=primary_test_labels,
        scores=primary_test_scores,
        frozen_threshold=primary_test_eval["threshold_congelado"],
        eer=primary_test_eval["EER"],
        title="Distribuição de Scores Genuínos vs. Impostores — CASIA-Iris-Interval",
        output_path=out_dir / "distributions.png",
    )

    logger.info("=" * 72)
    logger.info(
        f"[Experimento A - CASIA-Iris-Interval] "
        f"EER: {metrics_payload['EER_mean'] * 100:.2f}% ± {metrics_payload['EER_std'] * 100:.2f}% | "
        f"AUC: {metrics_payload['AUC_mean']:.4f} ± {metrics_payload['AUC_std']:.4f} | "
        f"FMR(τ*): {metrics_payload['FMR_mean'] * 100:.2f}% | FNMR(τ*): {metrics_payload['FNMR_mean'] * 100:.2f}% | "
        f"TAR@FAR=1%: {metrics_payload['TAR_at_FAR_1pct_mean'] * 100:.2f}%"
    )
    logger.info("=" * 72)

    # 5. Análise Automática de Erros (somente na avaliação principal com Rubber Sheet)
    if use_rs and results_subdir is None:
        run_error_analysis(
            scored_test_df=primary_test_scored_df,
            frozen_threshold=primary_test_eval["threshold_congelado"],
            output_csv=out_dir / "error_analysis.csv",
            output_png=out_dir / "error_analysis.png",
        )

    # 4. Experimento C - Cross-Dataset (CASIA -> IITD sem fine-tuning)
    cross_metrics_payload = None
    if has_cross and primary_cross_eval is not None and use_rs and results_subdir is None:
        cross_dir = out_dir / "cross_dataset_iitd"
        cross_dir.mkdir(parents=True, exist_ok=True)

        c_eers = [m["EER"] for m in iitd_seed_metrics]
        c_aucs = [m["AUC"] for m in iitd_seed_metrics]
        c_fmrs = [m["FMR_with_casia_threshold"] for m in iitd_seed_metrics]
        c_fnmrs = [m["FNMR_with_casia_threshold"] for m in iitd_seed_metrics]
        c_tars = [m["TAR_at_FAR_1pct"] for m in iitd_seed_metrics]

        cross_metrics_payload = {
            "experimento": "Experimento C — Cross-Dataset (Sem Fine-Tuning)",
            "modalidade": "Íris",
            "dataset_origem_treino": "CASIA-Iris-V3 Interval",
            "dataset_destino_avaliacao": "IIT Delhi Iris (IITD)",
            "cnn": "ResNet-50",
            "pre_treinamento": "ImageNet-1K (IMAGENET1K_V2)",
            "threshold_congelado_origem": round(float(primary_test_eval["threshold_congelado"]), 4),
            "EER_mean": round(float(np.mean(c_eers)), 4),
            "EER_std": round(float(np.std(c_eers)), 4),
            "AUC_mean": round(float(np.mean(c_aucs)), 4),
            "AUC_std": round(float(np.std(c_aucs)), 4),
            "FMR_mean": round(float(np.mean(c_fmrs)), 4),
            "FMR_std": round(float(np.std(c_fmrs)), 4),
            "FNMR_mean": round(float(np.mean(c_fnmrs)), 4),
            "FNMR_std": round(float(np.std(c_fnmrs)), 4),
            "TAR_at_FAR_1pct_mean": round(float(np.mean(c_tars)), 4),
            "TAR_at_FAR_1pct_std": round(float(np.std(c_tars)), 4),
            "tempo_inferencia_ms": metrics_payload["tempo_inferencia_ms"],
            "tamanho_modelo_mb": metrics_payload["tamanho_modelo_mb"],
            "hardware": metrics_payload["hardware"],
            "per_seed_results": iitd_seed_metrics,
        }

        save_scores_csv(primary_cross_scored_df, cross_dir / "scores.csv", is_casia=False)
        with open(cross_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(cross_metrics_payload, f, indent=2, ensure_ascii=False)

        plot_roc_curve(
            fpr=primary_cross_eval["fpr"],
            tpr=primary_cross_eval["tpr"],
            roc_auc=primary_cross_eval["AUC"],
            eer=primary_cross_eval["EER"],
            title="Curva ROC Cross-Dataset — ResNet-50 (Treino: CASIA → Teste: IITD)",
            output_path=cross_dir / "roc.png",
        )
        plot_score_distributions(
            labels=primary_cross_labels,
            scores=primary_cross_scores,
            frozen_threshold=primary_test_eval["threshold_congelado"],
            eer=primary_cross_eval["EER"],
            title="Distribuição de Scores Cross-Dataset — IITD (Threshold Congelado da CASIA)",
            output_path=cross_dir / "distributions.png",
        )

        logger.info("=" * 72)
        logger.info(
            f"[Experimento C - Cross-Dataset IITD] "
            f"EER: {cross_metrics_payload['EER_mean'] * 100:.2f}% ± {cross_metrics_payload['EER_std'] * 100:.2f}% | "
            f"AUC: {cross_metrics_payload['AUC_mean']:.4f} ± {cross_metrics_payload['AUC_std']:.4f} | "
            f"FMR(τ* CASIA): {cross_metrics_payload['FMR_mean'] * 100:.2f}% | "
            f"FNMR(τ* CASIA): {cross_metrics_payload['FNMR_mean'] * 100:.2f}%"
        )
        logger.info("=" * 72)

    return {
        "casia_metrics": metrics_payload,
        "cross_dataset_iitd_metrics": cross_metrics_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa avaliação biométrica 1:1 (Experimento A, Experimento C Cross-Dataset e Análise de Erros)."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho para config.yaml.")
    parser.add_argument(
        "--no-rubber-sheet",
        action="store_true",
        help="Avalia os modelos do Estudo de Ablação sem Rubber Sheet.",
    )
    args = parser.parse_args()

    if args.no_rubber_sheet:
        run_evaluation(
            config_path=args.config,
            use_rubber_sheet=False,
            results_subdir="results/ablation_no_rubbersheet",
        )
    else:
        run_evaluation(config_path=args.config, use_rubber_sheet=True)


if __name__ == "__main__":
    main()
