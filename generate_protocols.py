"""
Geração determinística de protocolos experimentais de verificação biométrica de íris 1:1.
Regras aplicadas:
1. Olho Esquerdo (L) != Olho Direito (R): identidades biométricas estritamente distintas.
2. Divisão mutuamente exclusiva por identidade (70% treino, 15% validação, 15% teste) na CASIA-Iris-Interval (seed=42).
3. Geração de pares genuínos e impostores (1:4) sem vazamento de identidade para Validação, Teste e Cross-Dataset (IITD).
"""

import argparse
import itertools
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png"}


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        path = Path(__file__).resolve().parent / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_casia_sample(img_path: Path, casia_root: Path) -> Optional[Tuple[str, str]]:
    """
    Extrai (identity_id, eye_side) de uma imagem da base CASIA-Iris-Interval.
    Exemplos suportados:
      - S1001L01.jpg -> ('CASIA_1001_L', 'L')
      - S1001R01.jpg -> ('CASIA_1001_R', 'R')
      - 001/L/S1001L01.jpg ou 001/L/01.jpg -> ('CASIA_001_L', 'L')
    """
    stem = img_path.stem

    # Padrão oficial CASIA-Iris-Interval: S1001L01.jpg
    match = re.match(r"^[sS]?(\d+)[_-]?([lLrR])\d+$", stem)
    if match:
        subj_id = match.group(1)
        eye_side = match.group(2).upper()
        return f"CASIA_{subj_id}_{eye_side}", eye_side

    # Padrão alternativo: busca lado do olho (L/R) e ID do sujeito nas pastas relativas
    rel_parts = img_path.relative_to(casia_root).parts
    eye_side = None
    subj_id = None

    for part in reversed(rel_parts[:-1]):
        part_upper = part.upper()
        if eye_side is None and part_upper in {"L", "R", "LEFT", "RIGHT"}:
            eye_side = "L" if part_upper.startswith("L") else "R"
        elif subj_id is None and re.match(r"^\d+$", part):
            subj_id = part

    if eye_side is None:
        if re.search(r"(_L|LEFT)", stem, re.IGNORECASE):
            eye_side = "L"
        elif re.search(r"(_R|RIGHT)", stem, re.IGNORECASE):
            eye_side = "R"

    if subj_id and eye_side:
        return f"CASIA_{subj_id}_{eye_side}", eye_side

    return None


def parse_iitd_sample(img_path: Path, iitd_root: Path) -> Optional[Tuple[str, str]]:
    """
    Extrai (identity_id, eye_side) de uma imagem da base IITD (sujeitos 001 a 224).
    Regras:
      1. Se arquivo/caminho contiver '_L' ou 'Left' -> IITD_{subj}_L
      2. Se arquivo/caminho contiver '_R' ou 'Right' -> IITD_{subj}_R
      3. Se for apenas numérico (01.bmp..10.bmp), amostras <= 5 são IITD_{subj}_L e >= 6 são IITD_{subj}_R.
    """
    rel_path = img_path.relative_to(iitd_root)
    rel_parts = rel_path.parts
    if any("normalized" in p.lower() for p in rel_parts):
        return None
    stem = img_path.stem

    # Identifica a pasta do sujeito (ex: 001 a 224)
    subj_id = None
    for part in rel_parts[:-1]:
        if re.match(r"^\d+$", part):
            subj_num = int(part)
            subj_id = f"{subj_num:03d}"
            break

    # Caso o nome do sujeito esteja codificado no próprio arquivo (ex: 001_01.bmp ou 001_1_L.bmp)
    if subj_id is None:
        m_prefix = re.match(r"^(\d+)[_-](\d+)", stem)
        if m_prefix:
            subj_id = f"{int(m_prefix.group(1)):03d}"

    if subj_id is None:
        return None

    rel_str = rel_path.as_posix()

    # Regra 1 e 2: Verifica _L / Left ou _R / Right no arquivo ou subcaminho relativo ao sujeito
    if re.search(r"(_L\b|left|/L/)", rel_str, re.IGNORECASE):
        eye_side = "L"
    elif re.search(r"(_R\b|right|/R/)", rel_str, re.IGNORECASE):
        eye_side = "R"
    else:
        # Regra 3: Apenas numérico (ex: 01.bmp..10.bmp) -> <= 5 é L, >= 6 é R
        nums = re.findall(r"\d+", stem)
        if not nums:
            return None
        sample_idx = int(nums[-1])
        eye_side = "L" if sample_idx <= 5 else "R"

    return f"IITD_{subj_id}_{eye_side}", eye_side


def collect_dataset_samples(
    root_dir: Path, dataset_name: str, is_casia: bool = True
) -> Dict[str, List[Dict[str, str]]]:
    """
    Varre o diretório do dataset, agrupa amostras por identity_id e descarta identidades
    com menos de 2 amostras.
    """
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Diretório da base '{dataset_name}' não encontrado em '{root_dir}'. "
            f"Consulte o README.md para organizar a base oficial localmente em 'data/'."
        )

    all_files = sorted(
        [
            p
            for p in root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda x: x.as_posix(),
    )

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    skipped_parse = 0

    for img_path in all_files:
        parsed = (
            parse_casia_sample(img_path, root_dir)
            if is_casia
            else parse_iitd_sample(img_path, root_dir)
        )
        if parsed is None:
            skipped_parse += 1
            continue

        identity_id, eye_side = parsed
        grouped[identity_id].append(
            {
                "image_path": img_path.as_posix(),
                "identity_id": identity_id,
                "dataset": dataset_name,
                "eye_side": eye_side,
            }
        )

    # Filtra identidades com menos de 2 amostras
    valid_grouped: Dict[str, List[Dict[str, str]]] = {}
    discarded_identities = 0
    for identity_id in sorted(grouped.keys()):
        samples = sorted(grouped[identity_id], key=lambda s: s["image_path"])
        if len(samples) >= 2:
            valid_grouped[identity_id] = samples
        else:
            discarded_identities += 1

    total_valid_samples = sum(len(v) for v in valid_grouped.values())
    logger.info(
        f"[{dataset_name}] Identidades válidas (>=2 amostras): {len(valid_grouped)} | "
        f"Total de amostras: {total_valid_samples} | "
        f"Identidades descartadas (<2 amostras): {discarded_identities}"
    )
    return valid_grouped


def split_identities(
    identities_dict: Dict[str, List[Dict[str, str]]],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Divide a lista de identidades únicas em 70% Treino, 15% Validação e 15% Teste
    de forma mutuamente exclusiva (Zero Identity Leakage).
    """
    unique_ids = sorted(identities_dict.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    train_ids = sorted(unique_ids[:n_train])
    val_ids = sorted(unique_ids[n_train : n_train + n_val])
    test_ids = sorted(unique_ids[n_train + n_val :])

    # Verificação estrita anti-Identity Leakage
    assert set(train_ids).isdisjoint(set(val_ids)), "ERRO: Vazamento de identidade entre Treino e Validação!"
    assert set(train_ids).isdisjoint(set(test_ids)), "ERRO: Vazamento de identidade entre Treino e Teste!"
    assert set(val_ids).isdisjoint(set(test_ids)), "ERRO: Vazamento de identidade entre Validação e Teste!"

    return train_ids, val_ids, test_ids


def build_split_dataframe(
    identities_dict: Dict[str, List[Dict[str, str]]], selected_ids: List[str]
) -> pd.DataFrame:
    rows = []
    for cid in selected_ids:
        rows.extend(identities_dict[cid])
    return pd.DataFrame(rows, columns=["image_path", "identity_id", "dataset", "eye_side"])


def generate_verification_pairs(
    identities_dict: Dict[str, List[Dict[str, str]]],
    selected_ids: List[str],
    impostor_ratio: int = 4,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Gera todos os pares genuínos possíveis dentro do conjunto de identidades selecionadas
    e amostra pares impostores na proporção 1:impostor_ratio com semente fixa.
    Retorna: (df_genuine, df_impostor, df_all) com colunas [amostra_1, amostra_2, tipo].
    """
    rng = random.Random(seed)
    selected_ids_sorted = sorted(selected_ids)

    # 1. Todos os pares genuínos (mesma identidade biométrica: mesmo sujeito E mesmo olho L/R)
    genuine_records = []
    for cid in selected_ids_sorted:
        img_paths = [s["image_path"] for s in identities_dict[cid]]
        for img1, img2 in itertools.combinations(img_paths, 2):
            genuine_records.append(
                {"amostra_1": img1, "amostra_2": img2, "tipo": "genuine"}
            )

    n_genuine = len(genuine_records)
    target_impostors = n_genuine * impostor_ratio

    # 2. Pares impostores (identidades biométricas distintas)
    impostor_records = []
    if len(selected_ids_sorted) >= 2 and target_impostors > 0:
        # Calcula máximo teórico de pares impostores
        counts = [len(identities_dict[cid]) for cid in selected_ids_sorted]
        total_imgs = sum(counts)
        max_possible_impostors = (
            total_imgs * total_imgs - sum(c * c for c in counts)
        ) // 2

        if max_possible_impostors <= target_impostors:
            # Se o número total possível for menor ou igual ao desejado, enumera todos
            for id1, id2 in itertools.combinations(selected_ids_sorted, 2):
                paths1 = [s["image_path"] for s in identities_dict[id1]]
                paths2 = [s["image_path"] for s in identities_dict[id2]]
                for p1 in paths1:
                    for p2 in paths2:
                        impostor_records.append(
                            {"amostra_1": p1, "amostra_2": p2, "tipo": "impostor"}
                        )
        else:
            seen_pairs = set()
            while len(impostor_records) < target_impostors:
                id1, id2 = rng.sample(selected_ids_sorted, 2)
                if id1 > id2:
                    id1, id2 = id2, id1
                p1 = rng.choice(identities_dict[id1])["image_path"]
                p2 = rng.choice(identities_dict[id2])["image_path"]
                pair_key = (p1, p2)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    impostor_records.append(
                        {"amostra_1": p1, "amostra_2": p2, "tipo": "impostor"}
                    )

    df_genuine = pd.DataFrame(genuine_records, columns=["amostra_1", "amostra_2", "tipo"])
    df_impostor = pd.DataFrame(impostor_records, columns=["amostra_1", "amostra_2", "tipo"])
    df_all = pd.concat([df_genuine, df_impostor], ignore_index=True)

    return df_genuine, df_impostor, df_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera os arquivos de protocolo (Treino/Val/Teste e Pares 1:1) com semente fixa."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho para o arquivo config.yaml.")
    parser.add_argument("--protocols-dir", type=str, default="protocols", help="Diretório de saída dos protocolos.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("protocol_seed", 42))
    impostor_ratio = int(cfg.get("impostor_ratio", 4))

    random.seed(seed)
    np.random.seed(seed)

    casia_dir = Path(cfg.get("casia_dir", "data/CASIA-Iris-Interval"))
    iitd_dir = Path(cfg.get("iitd_dir", "data/IITD"))
    protocols_dir = Path(args.protocols_dir)
    protocols_dir.mkdir(parents=True, exist_ok=True)

    # 1. Processa Base Principal (CASIA-Iris-Interval)
    logger.info(f"Processando Base Principal: CASIA-Iris-Interval (seed={seed})...")
    casia_identities = collect_dataset_samples(
        root_dir=casia_dir,
        dataset_name=cfg.get("primary_dataset", "CASIA-Iris-Interval"),
        is_casia=True,
    )

    train_ids, val_ids, test_ids = split_identities(
        casia_identities, seed=seed, train_ratio=0.70, val_ratio=0.15
    )

    df_train = build_split_dataframe(casia_identities, train_ids)
    df_val = build_split_dataframe(casia_identities, val_ids)
    df_test = build_split_dataframe(casia_identities, test_ids)

    df_train.to_csv(protocols_dir / "train.csv", index=False)
    df_val.to_csv(protocols_dir / "validation.csv", index=False)
    df_test.to_csv(protocols_dir / "test.csv", index=False)

    logger.info(
        f"[CASIA Split] Treino: {len(train_ids)} identidades ({len(df_train)} imagens) | "
        f"Validação: {len(val_ids)} identidades ({len(df_val)} imagens) | "
        f"Teste: {len(test_ids)} identidades ({len(df_test)} imagens)"
    )

    # 2. Formação dos pares da Validação da CASIA (1:4)
    val_gen, val_imp, val_pairs = generate_verification_pairs(
        casia_identities, val_ids, impostor_ratio=impostor_ratio, seed=seed
    )
    val_pairs.to_csv(protocols_dir / "val_pairs.csv", index=False)
    logger.info(
        f"[CASIA Validação Pares] Genuínos: {len(val_gen)} | Impostores: {len(val_imp)} | "
        f"Total salvo em 'protocols/val_pairs.csv': {len(val_pairs)}"
    )

    # 3. Formação dos pares do Teste da CASIA (1:4)
    test_gen, test_imp, test_pairs = generate_verification_pairs(
        casia_identities, test_ids, impostor_ratio=impostor_ratio, seed=seed
    )
    test_gen.to_csv(protocols_dir / "genuine_pairs.csv", index=False)
    test_imp.to_csv(protocols_dir / "impostor_pairs.csv", index=False)
    test_pairs.to_csv(protocols_dir / "test_pairs.csv", index=False)
    logger.info(
        f"[CASIA Teste Pares] Genuínos: {len(test_gen)} ('genuine_pairs.csv') | "
        f"Impostores: {len(test_imp)} ('impostor_pairs.csv') | "
        f"Total unificado ('test_pairs.csv'): {len(test_pairs)}"
    )

    # 4. Processa Segunda Base (IITD - Cross-Dataset)
    grupo6_iitd_pairs = Path("handoff_para_grupo5/protocols/test_pairs.csv")
    if grupo6_iitd_pairs.exists():
        logger.info(
            f"Importando protocolo oficial de pares de teste da IITD do Grupo 6 ('{grupo6_iitd_pairs.as_posix()}') "
            f"para garantir pares 100% idênticos entre os Grupos 5 e 6 (Seção 11)..."
        )
        df_g6 = pd.read_csv(grupo6_iitd_pairs)
        for col in ["amostra_1", "amostra_2"]:
            df_g6[col] = (
                df_g6[col]
                .astype(str)
                .str.replace("\\", "/", regex=False)
                .str.replace("./data/IITD_Iris/", f"{iitd_dir.as_posix()}/", regex=False)
                .str.replace("data/IITD_Iris/", f"{iitd_dir.as_posix()}/", regex=False)
            )
        df_g6.to_csv(protocols_dir / "cross_dataset_test_pairs.csv", index=False)
        n_gen_g6 = int((df_g6["tipo"] == "genuine").sum())
        n_imp_g6 = int((df_g6["tipo"] == "impostor").sum())
        logger.info(
            f"[IITD Cross-Dataset Pares Sincronizados com Grupo 6] "
            f"Genuínos: {n_gen_g6} | Impostores: {n_imp_g6} | "
            f"Total salvo em 'protocols/cross_dataset_test_pairs.csv': {len(df_g6)}"
        )
    elif iitd_dir.exists():
        logger.info(f"Processando Segunda Base (Cross-Dataset): IITD (seed={seed})...")
        iitd_identities = collect_dataset_samples(
            root_dir=iitd_dir,
            dataset_name=cfg.get("secondary_dataset", "IITD"),
            is_casia=False,
        )
        all_iitd_ids = sorted(iitd_identities.keys())
        iitd_gen, iitd_imp, iitd_pairs = generate_verification_pairs(
            iitd_identities, all_iitd_ids, impostor_ratio=impostor_ratio, seed=seed
        )
        iitd_pairs.to_csv(protocols_dir / "cross_dataset_test_pairs.csv", index=False)
        logger.info(
            f"[IITD Cross-Dataset Pares] Identidades: {len(all_iitd_ids)} | "
            f"Genuínos: {len(iitd_gen)} | Impostores: {len(iitd_imp)} | "
            f"Total salvo em 'protocols/cross_dataset_test_pairs.csv': {len(iitd_pairs)}"
        )
    else:
        logger.warning(
            f"Diretório IITD ('{iitd_dir}') não encontrado. "
            f"Organize a base oficial em '{iitd_dir}' conforme o README.md e rode 'generate_protocols.py' novamente."
        )


if __name__ == "__main__":
    main()
