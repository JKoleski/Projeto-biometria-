"""
Módulo de pré-processamento padronizado de Íris para a ResNet-50:
1. Controle de Qualidade (nitidez via variância do Laplaciano) e contabilização de FTA (Failure to Acquire).
2. Remoção de reflexos especulares via máscara binária (> 240) + cv2.inpaint.
3. Segmentação da Pupila (borda interna: xp, yp, rp) e do Limbo/Esclera (borda externa: xi, yi, ri) com fallback geométrico concêntrico (ri = 2.2 * rp).
4. Transformação Rubber Sheet de Daugman (64x512) + CLAHE + Redimensionamento (224x224) + Normalização ImageNet.
5. Suporte ao Estudo de Ablação (use_rubber_sheet=True vs. use_rubber_sheet=False).
6. Geração de exemplos visuais reais das etapas do pipeline em results/pipeline_examples.png.
"""

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png"}


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        path = Path(__file__).resolve().parent / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class SegmentationResult:
    """Armazena os parâmetros geométricos da segmentação pupila/limbo e metadados de qualidade."""
    xp: float
    yp: float
    rp: float
    xi: float
    yi: float
    ri: float
    used_limbus_fallback: bool = False
    used_pupil_fallback: bool = False
    laplacian_var: float = 0.0
    is_fta: bool = False
    fta_reason: Optional[str] = None


@dataclass
class FTATracker:
    """Registra ocorrências de Failure to Acquire (FTA) durante o pré-processamento."""
    total_evaluated: int = 0
    fta_count: int = 0
    limbus_fallback_count: int = 0
    pupil_fallback_count: int = 0
    failures: List[Dict[str, Union[str, float]]] = field(default_factory=list)

    def record(
        self,
        image_path: str,
        is_fta: bool,
        reason: Optional[str] = None,
        laplacian_var: float = 0.0,
        used_pupil_fallback: bool = False,
        used_limbus_fallback: bool = False,
    ) -> None:
        self.total_evaluated += 1
        if used_pupil_fallback:
            self.pupil_fallback_count += 1
        if used_limbus_fallback:
            self.limbus_fallback_count += 1
        if is_fta:
            self.fta_count += 1
            self.failures.append(
                {
                    "image_path": str(image_path),
                    "reason": reason or "unknown_error",
                    "laplacian_var": round(float(laplacian_var), 4),
                }
            )

    @property
    def fta_rate(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return self.fta_count / self.total_evaluated

    def to_dict(self) -> dict:
        return {
            "total_evaluated": self.total_evaluated,
            "fta_count": self.fta_count,
            "fta_rate": round(self.fta_rate, 6),
            "fta_percentage": round(self.fta_rate * 100.0, 4),
            "pupil_fallback_count": self.pupil_fallback_count,
            "limbus_fallback_count": self.limbus_fallback_count,
            "failures": self.failures,
        }

    def save_json(self, output_path: Union[str, Path] = "results/fta_report.json") -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def check_image_quality(
    gray_img: Optional[np.ndarray], min_laplacian_var: float = 3.0
) -> Tuple[bool, float, Optional[str]]:
    """
    1. Controle de Qualidade:
       Verifica se a imagem foi lida corretamente, possui dimensões mínimas
       e nitidez mínima (variância do Laplaciano > min_laplacian_var).
    """
    if gray_img is None or not isinstance(gray_img, np.ndarray) or gray_img.size == 0:
        return False, 0.0, "corrupted_or_unreadable_image"

    if gray_img.ndim != 2 or gray_img.shape[0] < 32 or gray_img.shape[1] < 32:
        return False, 0.0, f"invalid_dimensions_{gray_img.shape}"

    lap_var = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
    if not np.isfinite(lap_var) or lap_var < min_laplacian_var:
        return False, lap_var, f"low_sharpness_laplacian_var_{lap_var:.2f}"

    return True, lap_var, None


def remove_specular_reflections(gray_img: np.ndarray, threshold: int = 200) -> np.ndarray:
    """
    Remove reflexos especulares intensos (pixels > 200, incluindo halos de LEDs NIR na pupila)
    utilizando máscara binária dilatada + cv2.inpaint.
    """
    _, spec_mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(spec_mask) == 0:
        return gray_img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    spec_mask_dilated = cv2.dilate(spec_mask, kernel, iterations=2)
    inpainted = cv2.inpaint(gray_img, spec_mask_dilated, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    return inpainted


def localize_pupil(gray_inpainted: np.ndarray) -> Tuple[float, float, float, bool]:
    """
    Localiza a borda interna da íris (Pupila: xp, yp, rp):
    1. Suavização Gaussiana.
    2. Limiarização de Otsu/adaptativa invertida restrita à região central + abertura morfológica.
    3. Seleção do contorno circular mais consistente próximo ao centro ou cv2.HoughCircles.
    Retorna (xp, yp, rp, used_fallback).
    """
    h, w = gray_inpainted.shape[:2]
    min_dim = min(h, w)
    cx_img, cy_img = w * 0.5, h * 0.5

    min_rp = max(10, int(min_dim * 0.06))
    max_rp = int(min_dim * 0.28)

    blurred = cv2.GaussianBlur(gray_inpainted, (7, 7), 0)

    # Caso a imagem já possua máscara circular externa preta nos cantos (ex: IITD Segmented),
    # preenche o fundo externo com intensidade neutra para não interferir na limiarização da pupila central
    corner_mean = float(
        np.mean([blurred[0, 0], blurred[0, w - 1], blurred[h - 1, 0], blurred[h - 1, w - 1]])
    )
    yy_grid, xx_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((xx_grid - cx_img) ** 2 + (yy_grid - cy_img) ** 2)
    if corner_mean < 5.0:
        inner_vals = blurred[(dist_from_center < min_dim * 0.35) & (blurred > 10)]
        fill_val = int(np.median(inner_vals)) if inner_vals.size > 0 else 128
        blurred_for_pupil = blurred.copy()
        blurred_for_pupil[(blurred <= 5) & (dist_from_center > min_dim * 0.30)] = fill_val
    else:
        blurred_for_pupil = blurred

    # Limiarização na região central para evitar bordas escuras de cílios/cantos
    margin_y, margin_x = int(h * 0.18), int(w * 0.18)
    central_roi = blurred_for_pupil[margin_y : h - margin_y, margin_x : w - margin_x]
    otsu_thresh, _ = cv2.threshold(central_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    p20_thresh = float(np.percentile(central_roi, 20))
    effective_thresh = min(float(otsu_thresh), p20_thresh + 15.0)

    _, binary_inv = cv2.threshold(blurred_for_pupil, int(effective_thresh), 255, cv2.THRESH_BINARY_INV)

    # Restringe candidatos de pupila ao disco central (raio <= 0.34 * min_dim)
    mask_border = np.zeros_like(binary_inv)
    cv2.circle(
        mask_border,
        (int(round(cx_img)), int(round(cy_img))),
        int(round(min_dim * 0.34)),
        255,
        thickness=-1,
    )
    binary_inv = cv2.bitwise_and(binary_inv, mask_border)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    opened = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel_open, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best_score = -1.0
    best_circle: Optional[Tuple[float, float, float]] = None

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        if area < math.pi * (min_rp ** 2) * 0.5 or area > math.pi * (max_rp ** 2) * 1.4:
            continue

        perimeter = cv2.arcLength(hull, True)
        if perimeter <= 1e-5:
            continue

        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        if circularity < 0.55:
            continue

        (xc, yc), rc = cv2.minEnclosingCircle(hull)
        if not (min_rp <= rc <= max_rp):
            continue

        dist_center = math.hypot(xc - cx_img, yc - cy_img) / (min_dim * 0.5)
        if dist_center > 0.55:
            continue

        # Verifica se o interior do círculo candidato é realmente escuro (pupila real, não cílio)
        mask_p = np.zeros_like(blurred_for_pupil)
        cv2.circle(mask_p, (int(round(xc)), int(round(yc))), max(3, int(round(rc * 0.75))), 255, -1)
        mean_inner = float(cv2.mean(blurred_for_pupil, mask=mask_p)[0])
        if mean_inner > float(effective_thresh) + 25.0:
            continue

        # Score combinando circularidade, escuridão interna e proximidade ao centro da imagem
        darkness_bonus = max(0.0, (120.0 - mean_inner) / 120.0)
        score = circularity * 2.0 + darkness_bonus - dist_center * 0.9
        if score > best_score:
            best_score = score
            best_circle = (float(xc), float(yc), float(rc))

    if best_circle is not None:
        return best_circle[0], best_circle[1], best_circle[2], False

    # Fallback 1: Transformada de Hough Circular para a pupila
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim // 4,
        param1=70,
        param2=22,
        minRadius=min_rp,
        maxRadius=max_rp,
    )
    if circles is not None and len(circles) > 0:
        candidates = circles[0]
        best_hough = None
        best_hough_score = float("inf")
        for c in candidates:
            xc, yc, rc = float(c[0]), float(c[1]), float(c[2])
            dist = math.hypot(xc - cx_img, yc - cy_img)
            if dist < min_dim * 0.35:
                # Prefere candidato escuro no interior e próximo ao centro
                mask_p = np.zeros_like(gray_inpainted)
                cv2.circle(mask_p, (int(round(xc)), int(round(yc))), int(round(rc * 0.8)), 255, -1)
                mean_intensity = cv2.mean(blurred, mask=mask_p)[0]
                h_score = mean_intensity + 0.5 * dist
                if h_score < best_hough_score:
                    best_hough_score = h_score
                    best_hough = (xc, yc, rc)
        if best_hough is not None:
            return best_hough[0], best_hough[1], best_hough[2], False

    # Fallback geométrico central caso nem contorno nem Hough encontrem a pupila
    return float(cx_img), float(cy_img), float(min_dim * 0.14), True


def localize_limbus(
    gray_inpainted: np.ndarray, xp: float, yp: float, rp: float
) -> Tuple[float, float, float, bool]:
    """
    Localiza a borda externa da íris (Limbo/Esclera: xi, yi, ri):
    - Aplica suavização + Canny/gradiente + cv2.HoughCircles restrito a raios ri em [1.5 * rp, 3.2 * rp]
      e centro próximo a (xp, yp).
    - Caso o Hough não encontre círculo válido em amostra difícil, aplica fallback geométrico
      concêntrico a (xp, yp) com ri = 2.2 * rp.
    """
    h, w = gray_inpainted.shape[:2]
    min_dim = min(h, w)

    min_ri = max(int(round(1.85 * rp)), int(min_dim * 0.28))
    max_ri = min(int(round(3.2 * rp)), int(min_dim * 0.48))
    if max_ri <= min_ri:
        max_ri = min_ri + max(10, int(rp * 0.6))

    # Suavização bilateral/gaussiana preservando a transição limbo-esclera
    blurred = cv2.GaussianBlur(gray_inpainted, (9, 9), 1.5)
    clahe_temp = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)

    # 1. Tentativa via cv2.HoughCircles (usa Canny internamente no HOUGH_GRADIENT)
    max_center_offset = max(12.0, 0.40 * rp)

    for param2 in (32, 24, 18):
        circles = cv2.HoughCircles(
            clahe_temp,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim // 4,
            param1=80,
            param2=param2,
            minRadius=min_ri,
            maxRadius=max_ri,
        )
        if circles is None or len(circles) == 0:
            continue

        valid_candidates = []
        for c in circles[0]:
            xi_c, yi_c, ri_c = float(c[0]), float(c[1]), float(c[2])
            center_dist = math.hypot(xi_c - xp, yi_c - yp)
            if center_dist <= max_center_offset and (1.5 * rp <= ri_c <= 3.2 * rp):
                valid_candidates.append((center_dist, xi_c, yi_c, ri_c))

        if valid_candidates:
            # Seleciona o candidato com centro mais próximo de (xp, yp)
            valid_candidates.sort(key=lambda item: item[0])
            _, xi_best, yi_best, ri_best = valid_candidates[0]
            return xi_best, yi_best, ri_best, False

    # 2. Busca por maximização de gradiente radial de Daugman em torno de (xp, yp) no intervalo [1.5*rp, 3.2*rp]
    sobel_x = cv2.Sobel(clahe_temp, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(clahe_temp, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobel_x, sobel_y)

    # Avalia arcos laterais esquerdo e direito (-45° a +45° e 135° a 225°) para evitar oclusão de pálpebras superior/inferior
    angles = np.concatenate(
        [
            np.linspace(-math.pi / 4.0, math.pi / 4.0, 36),
            np.linspace(3.0 * math.pi / 4.0, 5.0 * math.pi / 4.0, 36),
        ]
    )
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)

    best_grad = -1.0
    best_r_grad: Optional[float] = None

    for r_cand in range(min_ri, max_ri + 1, 2):
        xs = np.clip(np.round(xp + r_cand * cos_t).astype(int), 0, w - 1)
        ys = np.clip(np.round(yp + r_cand * sin_t).astype(int), 0, h - 1)
        mean_g = float(np.mean(grad_mag[ys, xs]))
        if mean_g > best_grad:
            best_grad = mean_g
            best_r_grad = float(r_cand)

    if best_r_grad is not None and best_grad > 12.0:
        return float(xp), float(yp), best_r_grad, False

    # 3. Fallback geométrico concêntrico obrigatório: (xp, yp) com ri = 2.2 * rp
    ri_fallback = float(np.clip(2.2 * rp, min_ri, max(min_ri + 5, int(min_dim * 0.46))))
    return float(xp), float(yp), ri_fallback, True


def segment_iris(
    gray_img: np.ndarray, min_laplacian_var: float = 3.0
) -> Tuple[np.ndarray, SegmentationResult]:
    """
    Executa controle de qualidade, remoção de reflexos especulares e segmentação Pupila/Limbo.
    Retorna (gray_inpainted, SegmentationResult).
    """
    ok, lap_var, reason = check_image_quality(gray_img, min_laplacian_var=min_laplacian_var)
    if not ok:
        # Mesmo se falhar no controle de qualidade (FTA), retorna geometria central segura para não quebrar pipelines em lote
        if gray_img is not None and gray_img.ndim == 2 and gray_img.size > 0:
            h, w = gray_img.shape[:2]
            cx, cy = w * 0.5, h * 0.5
            rp = min(h, w) * 0.14
            ri = rp * 2.2
            inpainted = gray_img.copy()
        else:
            inpainted = np.zeros((224, 224), dtype=np.uint8)
            cx, cy, rp, ri = 112.0, 112.0, 30.0, 66.0

        return inpainted, SegmentationResult(
            xp=cx,
            yp=cy,
            rp=rp,
            xi=cx,
            yi=cy,
            ri=ri,
            used_limbus_fallback=True,
            used_pupil_fallback=True,
            laplacian_var=lap_var,
            is_fta=True,
            fta_reason=reason,
        )

    inpainted = remove_specular_reflections(gray_img, threshold=200)
    xp, yp, rp, pupil_fb = localize_pupil(inpainted)
    xi, yi, ri, limbus_fb = localize_limbus(inpainted, xp, yp, rp)

    return inpainted, SegmentationResult(
        xp=xp,
        yp=yp,
        rp=rp,
        xi=xi,
        yi=yi,
        ri=ri,
        used_limbus_fallback=limbus_fb,
        used_pupil_fallback=pupil_fb,
        laplacian_var=lap_var,
        is_fta=False,
        fta_reason=None,
    )


def daugman_rubber_sheet(
    gray_img: np.ndarray,
    seg: SegmentationResult,
    polar_height: int = 64,
    polar_width: int = 512,
) -> np.ndarray:
    """
    3. Modelo Rubber Sheet de Daugman:
       Mapeia o anel entre a pupila (xp, yp, rp) e o limbo (xi, yi, ri) de coordenadas
       cartesianas (x, y) para coordenadas polares adimensionais (r, theta) com dimensões
       (polar_height=64, polar_width=512).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, polar_width, endpoint=False, dtype=np.float32)
    r = np.linspace(0.0, 1.0, polar_height, endpoint=True, dtype=np.float32)[:, np.newaxis]

    cos_t = np.cos(theta)[np.newaxis, :]
    sin_t = np.sin(theta)[np.newaxis, :]

    # Pontos na borda da pupila (r = 0)
    xp_theta = seg.xp + seg.rp * cos_t
    yp_theta = seg.yp + seg.rp * sin_t

    # Pontos na borda do limbo (r = 1)
    xi_theta = seg.xi + seg.ri * cos_t
    yi_theta = seg.yi + seg.ri * sin_t

    # Interpolação linear radial para r in [0, 1]
    map_x = ((1.0 - r) * xp_theta + r * xi_theta).astype(np.float32)
    map_y = ((1.0 - r) * yp_theta + r * yi_theta).astype(np.float32)

    polar_iris = cv2.remap(
        gray_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return polar_iris


def crop_iris_roi_ablation(gray_img: np.ndarray, seg: SegmentationResult) -> np.ndarray:
    """
    4. Modo de Estudo de Ablação (use_rubber_sheet = False):
       Recorta diretamente o bounding box quadrado da íris (xi +- ri, yi +- ri) sem desenrolar
       em coordenadas polares e aplica máscara circular externa.
    """
    h, w = gray_img.shape[:2]
    ri_int = max(16, int(round(seg.ri)))
    cx_int = int(round(seg.xi))
    cy_int = int(round(seg.yi))

    # Adiciona padding caso o bounding box (xi +- ri, yi +- ri) ultrapasse as bordas da imagem
    pad = max(0, ri_int - cx_int, ri_int - cy_int, (cx_int + ri_int) - w, (cy_int + ri_int) - h)
    if pad > 0:
        padded = cv2.copyMakeBorder(gray_img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        cx_int += pad
        cy_int += pad
    else:
        padded = gray_img

    x1, x2 = cx_int - ri_int, cx_int + ri_int
    y1, y2 = cy_int - ri_int, cy_int + ri_int
    roi = padded[y1:y2, x1:x2].copy()

    # Aplica máscara circular externa
    roi_h, roi_w = roi.shape[:2]
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.circle(mask, (roi_w // 2, roi_h // 2), min(roi_h, roi_w) // 2, 255, thickness=-1)
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
    return masked_roi


def apply_clahe(img_uint8: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Aplica equalização de histograma adaptativa limitada por contraste (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_uint8)


def to_imagenet_tensor(img_224_gray: np.ndarray) -> torch.Tensor:
    """
    Converte imagem 224x224 uint8 em escala de cinza para tensor PyTorch (3, 224, 224)
    normalizado com mean=[0.485, 0.456, 0.406] e std=[0.229, 0.224, 0.225].
    """
    img_float = img_224_gray.astype(np.float32) / 255.0
    img_rgb = np.stack([img_float, img_float, img_float], axis=-1)  # (H, W, 3)
    img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()  # (3, H, W)
    return tensor


class IrisPreprocessor:
    """
    Pipeline completo e configurável de pré-processamento de íris para a ResNet-50.
    Suporta modo padrão (Com Rubber Sheet) e modo de Estudo de Ablação (Sem Rubber Sheet),
    além de rastreamento automático de FTA (Failure to Acquire).
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        polar_height: int = 64,
        polar_width: int = 512,
        use_rubber_sheet: bool = True,
        min_laplacian_var: float = 3.0,
        perturbation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.polar_height = int(polar_height)
        self.polar_width = int(polar_width)
        self.use_rubber_sheet = bool(use_rubber_sheet)
        self.min_laplacian_var = float(min_laplacian_var)
        self.perturbation_fn = perturbation_fn
        self.fta_tracker = FTATracker()

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        use_rubber_sheet: Optional[bool] = None,
        perturbation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> "IrisPreprocessor":
        img_sz = cfg.get("image_size", [224, 224])
        use_rs = cfg.get("use_rubber_sheet", True) if use_rubber_sheet is None else use_rubber_sheet
        return cls(
            image_size=(img_sz[0], img_sz[1]),
            polar_height=cfg.get("polar_height", 64),
            polar_width=cfg.get("polar_width", 512),
            use_rubber_sheet=use_rs,
            min_laplacian_var=cfg.get("min_laplacian_var", 3.0),
            perturbation_fn=perturbation_fn,
        )

    def process_Steps(
        self, image_or_path: Union[str, Path, np.ndarray], track_fta: bool = True
    ) -> Dict[str, Union[np.ndarray, torch.Tensor, SegmentationResult]]:
        """
        Executa todas as etapas do pipeline e retorna os resultados intermediários e finais:
        - raw_gray: Imagem original em escala de cinza
        - segmentation: Objeto SegmentationResult (xp, yp, rp, xi, yi, ri, flags)
        - unwrapped_or_roi: Faixa Rubber Sheet (64x512) ou ROI quadrada (Ablação)
        - clahe_resized: Imagem 224x224 pós-CLAHE (uint8)
        - tensor: Tensor (3, 224, 224) normalizado padrão ImageNet
        """
        img_label = str(image_or_path) if isinstance(image_or_path, (str, Path)) else "memory_array"

        if isinstance(image_or_path, (str, Path)) and self.perturbation_fn is None:
            cache_key = (
                str(image_or_path),
                self.use_rubber_sheet,
                self.polar_height,
                self.polar_width,
            )
            if cache_key in _UNWRAPPED_MEMORY_CACHE:
                seg, unwrapped_or_roi = _UNWRAPPED_MEMORY_CACHE[cache_key]
                raw_gray = None
                inpainted = None
            else:
                raw_gray = cv2.imread(str(image_or_path), cv2.IMREAD_GRAYSCALE)
                inpainted, seg = segment_iris(raw_gray, min_laplacian_var=self.min_laplacian_var)
                if self.use_rubber_sheet:
                    unwrapped_or_roi = daugman_rubber_sheet(
                        inpainted,
                        seg,
                        polar_height=self.polar_height,
                        polar_width=self.polar_width,
                    )
                else:
                    unwrapped_or_roi = crop_iris_roi_ablation(inpainted, seg)
                _UNWRAPPED_MEMORY_CACHE[cache_key] = (seg, unwrapped_or_roi)
        else:
            if isinstance(image_or_path, (str, Path)):
                raw_gray = cv2.imread(str(image_or_path), cv2.IMREAD_GRAYSCALE)
            else:
                raw_gray = image_or_path
                if raw_gray is not None and raw_gray.ndim == 3:
                    raw_gray = cv2.cvtColor(raw_gray, cv2.COLOR_BGR2GRAY)

            if raw_gray is not None and self.perturbation_fn is not None:
                raw_gray = self.perturbation_fn(raw_gray)

            inpainted, seg = segment_iris(raw_gray, min_laplacian_var=self.min_laplacian_var)
            if self.use_rubber_sheet:
                unwrapped_or_roi = daugman_rubber_sheet(
                    inpainted,
                    seg,
                    polar_height=self.polar_height,
                    polar_width=self.polar_width,
                )
            else:
                unwrapped_or_roi = crop_iris_roi_ablation(inpainted, seg)

        if track_fta:
            self.fta_tracker.record(
                image_path=img_label,
                is_fta=seg.is_fta,
                reason=seg.fta_reason,
                laplacian_var=seg.laplacian_var,
                used_pupil_fallback=seg.used_pupil_fallback,
                used_limbus_fallback=seg.used_limbus_fallback,
            )

        clahe_img = apply_clahe(unwrapped_or_roi, clip_limit=2.0, tile_grid_size=(8, 8))
        clahe_resized = cv2.resize(
            clahe_img,
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = to_imagenet_tensor(clahe_resized)

        if raw_gray is None and isinstance(image_or_path, (str, Path)):
            raw_gray = unwrapped_or_roi
            inpainted = unwrapped_or_roi

        return {
            "raw_gray": raw_gray if raw_gray is not None else inpainted,
            "inpainted": inpainted,
            "segmentation": seg,
            "unwrapped_or_roi": unwrapped_or_roi,
            "clahe_resized": clahe_resized,
            "tensor": tensor,
        }

    def __call__(
        self, image_or_path: Union[str, Path, np.ndarray], track_fta: bool = True
    ) -> torch.Tensor:
        """Retorna diretamente o tensor (3, 224, 224) normalizado para entrada na ResNet-50."""
        return self.process_Steps(image_or_path, track_fta=track_fta)["tensor"]


_UNWRAPPED_MEMORY_CACHE: Dict[Tuple[str, bool, int, int], Tuple[SegmentationResult, np.ndarray]] = {}


class IrisDataset(Dataset):
    """
    Dataset PyTorch para treinamento e extração de embeddings de Íris.
    Suporta data augmentation específica para íris (translação circular horizontal / rotação ocular
    na faixa polar ou leve jitter fotométrico) quando is_train=True.
    """

    def __init__(
        self,
        csv_path_or_df: Union[str, Path, pd.DataFrame],
        preprocessor: IrisPreprocessor,
        is_train: bool = False,
        label_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        if isinstance(csv_path_or_df, (str, Path)):
            self.df = pd.read_csv(csv_path_or_df)
        else:
            self.df = csv_path_or_df.reset_index(drop=True)

        self.preprocessor = preprocessor
        self.is_train = is_train

        self.image_paths: List[str] = self.df["image_path"].astype(str).tolist()
        if "identity_id" in self.df.columns:
            self.identities: List[str] = self.df["identity_id"].astype(str).tolist()
            if label_to_idx is None:
                unique_ids = sorted(set(self.identities))
                self.label_to_idx = {cid: idx for idx, cid in enumerate(unique_ids)}
            else:
                self.label_to_idx = label_to_idx
            self.labels = [self.label_to_idx.get(cid, -1) for cid in self.identities]
        else:
            self.identities = ["" for _ in self.image_paths]
            self.label_to_idx = {}
            self.labels = [-1 for _ in self.image_paths]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        steps = self.preprocessor.process_Steps(img_path, track_fta=not self.is_train)
        unwrapped_or_roi = steps["unwrapped_or_roi"]

        if self.is_train:
            # Data augmentation biologicamente consistente para íris:
            # Pequeno shift circular horizontal na faixa polar equivale a leve rotação torsional do olho (+/- 2.8 graus)
            if self.preprocessor.use_rubber_sheet:
                max_shift = max(2, int(self.preprocessor.polar_width * 0.008))
                shift = int(np.random.randint(-max_shift, max_shift + 1))
                aug_roi = np.roll(unwrapped_or_roi, shift=shift, axis=1)
            else:
                aug_roi = unwrapped_or_roi

            clahe_img = apply_clahe(aug_roi, clip_limit=2.0, tile_grid_size=(8, 8))
            clahe_resized = cv2.resize(
                clahe_img,
                (self.preprocessor.image_size[1], self.preprocessor.image_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            tensor = to_imagenet_tensor(clahe_resized)
        else:
            tensor = steps["tensor"]

        return tensor, label, img_path


def draw_segmentation_overlay(gray_img: np.ndarray, seg: SegmentationResult) -> np.ndarray:
    """
    Desenha a borda da pupila (verde) e a borda do limbo (vermelho/laranja) sobre a imagem original
    para visualização no artigo científico.
    """
    rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # Pupila (verde)
    cv2.circle(
        rgb,
        (int(round(seg.xp)), int(round(seg.yp))),
        int(round(seg.rp)),
        (0, 230, 118),
        2,
    )
    cv2.circle(rgb, (int(round(seg.xp)), int(round(seg.yp))), 2, (0, 230, 118), -1)

    # Limbo (azul/ciano ou laranja vivo em RGB)
    limbus_color = (255, 145, 0) if not seg.used_limbus_fallback else (255, 214, 0)
    cv2.circle(
        rgb,
        (int(round(seg.xi)), int(round(seg.yi))),
        int(round(seg.ri)),
        limbus_color,
        2,
    )
    cv2.circle(rgb, (int(round(seg.xi)), int(round(seg.yi))), 2, limbus_color, -1)
    return rgb


def _select_sample_paths(cfg: dict, n_casia: int = 4, n_iitd: int = 2) -> List[Tuple[str, Path]]:
    """
    Seleciona 4 amostras reais da CASIA-Iris-Interval e 2 da IITD (priorizando arquivos dos protocolos
    ou buscando diretamente nos diretórios de dados).
    """
    selected: List[Tuple[str, Path]] = []

    casia_dir = Path(cfg.get("casia_dir", "data/CASIA-Iris-Interval"))
    iitd_dir = Path(cfg.get("iitd_dir", "data/IITD"))

    # Busca 4 amostras de sujeitos distintos da CASIA
    test_csv = Path("protocols/test.csv")
    casia_candidates: List[Path] = []
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        for _, group in df_test.groupby("identity_id"):
            p = Path(group.iloc[0]["image_path"])
            if p.exists():
                casia_candidates.append(p)
            if len(casia_candidates) >= n_casia:
                break

    if len(casia_candidates) < n_casia and casia_dir.exists():
        all_casia = sorted(
            [p for p in casia_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        step = max(1, len(all_casia) // max(1, n_casia))
        for p in all_casia[::step]:
            if p not in casia_candidates:
                casia_candidates.append(p)
            if len(casia_candidates) >= n_casia:
                break

    for p in casia_candidates[:n_casia]:
        selected.append((f"CASIA ({p.name})", p))

    # Busca 2 amostras de sujeitos distintos da IITD
    cross_csv = Path("protocols/cross_dataset_test_pairs.csv")
    iitd_candidates: List[Path] = []
    if cross_csv.exists():
        df_cross = pd.read_csv(cross_csv)
        seen_subjs = set()
        for img_str in df_cross["amostra_1"].tolist():
            p = Path(img_str)
            subj_key = p.stem.split("_")[0] if "_" in p.stem else p.parent.name
            if p.exists() and subj_key not in seen_subjs:
                seen_subjs.add(subj_key)
                iitd_candidates.append(p)
            if len(iitd_candidates) >= n_iitd:
                break

    if len(iitd_candidates) < n_iitd and iitd_dir.exists():
        all_iitd = sorted(
            [p for p in iitd_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        step = max(1, len(all_iitd) // max(1, n_iitd))
        for p in all_iitd[::step]:
            if p not in iitd_candidates:
                iitd_candidates.append(p)
            if len(iitd_candidates) >= n_iitd:
                break

    for p in iitd_candidates[:n_iitd]:
        selected.append((f"IITD ({p.parent.name}/{p.name})", p))

    return selected


def save_pipeline_examples(
    config_path: str = "config.yaml",
    output_path: str = "results/pipeline_examples.png",
) -> Optional[Path]:
    """
    5. Exemplos Reais do Pipeline para o Artigo:
       Seleciona 4 amostras reais da CASIA-Iris-Interval e 2 da IITD e salva em
       results/pipeline_examples.png uma figura com 4 colunas:
       (1) Dado Bruto, (2) Segmentação Pupila/Limbo sobreposta,
       (3) Faixa Rubber Sheet (64x512) / ROI, e (4) Imagem Normalizada pós-CLAHE (224x224).
    """
    cfg = load_config(config_path)
    samples = _select_sample_paths(cfg, n_casia=4, n_iitd=2)

    if not samples:
        logger.warning(
            "Nenhuma imagem encontrada em data/CASIA-Iris-Interval ou data/IITD para gerar 'pipeline_examples.png'."
        )
        return None

    preprocessor = IrisPreprocessor.from_config(cfg)
    n_rows = len(samples)

    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 2.7 * n_rows), dpi=200)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "(1) Dado Bruto (NIR)",
        "(2) Segmentação Pupila/Limbo",
        "(3) Faixa Rubber Sheet (64x512)" if preprocessor.use_rubber_sheet else "(3) ROI Íris Mascarada",
        "(4) Normalizada pós-CLAHE (224x224)",
    ]

    for row_idx, (label, img_path) in enumerate(samples):
        steps = preprocessor.process_Steps(img_path, track_fta=True)
        raw_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw_gray is None:
            raw_gray = steps["raw_gray"]
        seg: SegmentationResult = steps["segmentation"]
        unwrapped_or_roi = steps["unwrapped_or_roi"]
        clahe_resized = steps["clahe_resized"]
        overlay_rgb = draw_segmentation_overlay(raw_gray, seg)

        # Coluna 1: Dado Bruto
        ax0 = axes[row_idx, 0]
        ax0.imshow(raw_gray, cmap="gray")
        ax0.set_ylabel(label, fontsize=9, fontweight="bold")
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Coluna 2: Segmentação Pupila/Limbo sobreposta
        ax1 = axes[row_idx, 1]
        ax1.imshow(overlay_rgb)
        ax1.set_xlabel(
            f"Pupila r={seg.rp:.1f}px | Limbo r={seg.ri:.1f}px", fontsize=8
        )
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Coluna 3: Faixa Rubber Sheet (64x512) / ROI
        ax2 = axes[row_idx, 2]
        ax2.imshow(unwrapped_or_roi, cmap="gray", aspect="auto" if preprocessor.use_rubber_sheet else "equal")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Coluna 4: Imagem Normalizada pós-CLAHE (224x224)
        ax3 = axes[row_idx, 3]
        ax3.imshow(clahe_resized, cmap="gray")
        ax3.set_xticks([])
        ax3.set_yticks([])

        if row_idx == 0:
            for col_idx, title in enumerate(col_titles):
                axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold", pad=8)

    plt.suptitle(
        "Etapas do Pipeline de Pré-processamento de Íris (CASIA-Iris-Interval & IITD)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Figura de exemplos do pipeline salva com sucesso em '{out_file.as_posix()}'.")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa validação de pré-processamento e gera figura de exemplos.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho para config.yaml.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/pipeline_examples.png",
        help="Caminho de saída para a figura de exemplos do pipeline.",
    )
    args = parser.parse_args()

    save_pipeline_examples(config_path=args.config, output_path=args.output)


if __name__ == "__main__":
    main()
