"""
Script orquestrador de ponta a ponta para reprodução integral de todos os experimentos do artigo:
1. Verificação local das bases oficiais em data/ (CASIA-Iris-Interval e IITD)
2. Geração determinística dos protocolos sem Identity Leakage (seed=42, Olho L != R) -> generate_protocols.py
3. Geração da prancha de exemplos reais das etapas do pipeline -> preprocess.py
4. Treinamento da ResNet-50 com CosFace nas 3 seeds ([42, 123, 2026]) -> train.py
5. Extração de embeddings L2 e benchmark de latência/hardware -> extract_embeddings.py
6. Avaliação no Teste da CASIA (Experimento A), Cross-Dataset na IITD (Experimento C) e Análise de Erros -> evaluate.py
7. Experimento D — Teste de Robustez em 4 Níveis (Blur Gaussiano e Oclusão Biométrica) -> robustness.py
8. Estudo de Ablação (Com Rubber Sheet vs. Sem Rubber Sheet) -> robustness.py
"""

import argparse
import json
import logging
import time
from pathlib import Path

from evaluate import run_evaluation
from extract_embeddings import run_extraction
from generate_protocols import main as protocols_main
from preprocess import IMAGE_EXTENSIONS, load_config, save_pipeline_examples
from robustness import run_ablation_study, run_robustness_experiment
from train import run_training

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def verify_local_datasets(cfg: dict) -> None:
    """
    Verifica a presença e a organização local das bases oficiais em data/CASIA-Iris-Interval e data/IITD.
    """
    casia_dir = Path(cfg.get("casia_dir", "data/CASIA-Iris-Interval"))
    iitd_dir = Path(cfg.get("iitd_dir", "data/IITD"))

    for name, d in [
        ("CASIA-Iris-V3 Interval (Base Principal)", casia_dir),
        ("IIT Delhi Iris v1.0 (Segunda Base / Cross-Dataset)", iitd_dir),
    ]:
        if not d.exists():
            raise FileNotFoundError(
                f"Diretório da base oficial '{name}' não encontrado em '{d.as_posix()}'. "
                f"Consulte o README.md para instruções de obtenção oficial e organização local em 'data/'."
            )
        imgs = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        folders = [p for p in d.rglob("*") if p.is_dir()]
        if not imgs:
            raise RuntimeError(
                f"Nenhuma imagem encontrada em '{d.as_posix()}' para a base '{name}'. "
                f"Verifique a organização local descrita no README.md."
            )
        logger.info(
            f"[Base Oficial Verificada] {name}: {len(imgs)} imagens em {len(folders)} subpastas ('{d.as_posix()}')"
        )


def generate_markdown_summary_table(results_dir: Path = Path("results")) -> None:
    """
    Gera results/final_summary_table.md no formato exato da Tabela Padronizada (Seção 26 do PDF)
    para inclusão direta no artigo científico.
    """
    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        return

    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    cross_file = results_dir / "cross_dataset_iitd" / "metrics.json"
    cross_m = None
    if cross_file.exists():
        with open(cross_file, "r", encoding="utf-8") as f:
            cross_m = json.load(f)

    lines = [
        "# Tabela Padronizada de Resultados — Grupo 5 (Íris / ResNet-50)\n",
        "## 1. Experimento A — Base Principal (`CASIA-Iris-V3 Interval`)\n",
        "| Informação | Resultado |",
        "| :--- | :--- |",
        f"| **Modalidade** | {m['modalidade']} |",
        f"| **Dataset** | {m['dataset_principal']} |",
        f"| **CNN** | {m['cnn']} |",
        f"| **Pré-treinamento** | {m['pre_treinamento']} |",
        f"| **Camadas Congeladas** | `{', '.join(m['camadas_congeladas'])}` |",
        f"| **Camadas Treinadas** | `{', '.join(m['camadas_treinadas'])}` |",
        f"| **Função de Perda** | {m['funcao_perda']} |",
        f"| **Nº de parâmetros (Total / Treináveis)** | {m['num_parametros_total']:,} / {m['num_parametros_treinaveis']:,} |",
        f"| **Dimensão do embedding** | {m['dimensao_embedding']} (L2-normalizado) |",
        f"| **Matching** | {m['matching']} |",
        f"| **Threshold Congelado (`τ*`)** | {m['threshold_congelado']:.4f} |",
        f"| **EER (Média ± DP)** | {m['EER_mean'] * 100:.2f}% ± {m['EER_std'] * 100:.2f}% |",
        f"| **AUC (Média ± DP)** | {m['AUC_mean']:.4f} ± {m['AUC_std']:.4f} |",
        f"| **FMR / FAR (`τ*`)** | {m['FMR_mean'] * 100:.2f}% ± {m['FMR_std'] * 100:.2f}% |",
        f"| **FNMR / FRR (`τ*`)** | {m['FNMR_mean'] * 100:.2f}% ± {m['FNMR_std'] * 100:.2f}% |",
        f"| **TAR@FAR=1%** | {m['TAR_at_FAR_1pct_mean'] * 100:.2f}% ± {m['TAR_at_FAR_1pct_std'] * 100:.2f}% |",
        f"| **FTA (Failure to Acquire)** | {m['FTA_rate'] * 100:.2f}% |",
        f"| **Tempo médio de inferência** | {m['tempo_inferencia_ms']:.3f} ms / amostra |",
        f"| **Tamanho do modelo** | {m['tamanho_modelo_mb']:.2f} MB |",
        f"| **Hardware** | {m['hardware']} |",
    ]

    if cross_m is not None:
        lines.extend(
            [
                "\n## 2. Experimento B — Matriz 2 CNNs x 2 Datasets (Modalidade Íris)\n",
                "| Arquitetura CNN (Grupo) | Base A: `CASIA-Iris-V3 Interval` (EER / AUC) | Base B: `IIT Delhi Iris v1.0` (EER / AUC) | Nº de Parâmetros | Tamanho (MB) | Inferência (ms) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
                f"| **ResNet-50 (Grupo 5 — Nosso)** | **EER: {m['EER_mean'] * 100:.2f}% ± {m['EER_std'] * 100:.2f}%**<br>**AUC: {m['AUC_mean']:.4f} ± {m['AUC_std']:.4f}** | **EER: {cross_m['EER_mean'] * 100:.2f}% ± {cross_m['EER_std'] * 100:.2f}%**<br>**AUC: {cross_m['AUC_mean']:.4f} ± {cross_m['AUC_std']:.4f}** | {m['num_parametros_total']:,} | {m['tamanho_modelo_mb']:.2f} MB | {m['tempo_inferencia_ms']:.3f} ms |",
                "| **CNN do Grupo 6** *(a consolidar com o Grupo 6)* | *Inserir EER/AUC do Grupo 6 na CASIA* | *Inserir EER/AUC do Grupo 6 na IITD* | *-* | *-* | *-* |",
                "\n## 3. Experimento C — Cross-Dataset (`CASIA-Iris-Interval` $\\rightarrow$ `IIT Delhi Iris` sem Fine-Tuning)\n",
                "| Métrica | Resultado (Média ± DP) |",
                "| :--- | :--- |",
                f"| **Threshold Congelado da Origem (`τ*`)** | {cross_m['threshold_congelado_origem']:.4f} |",
                f"| **EER Cross-Dataset** | {cross_m['EER_mean'] * 100:.2f}% ± {cross_m['EER_std'] * 100:.2f}% |",
                f"| **AUC Cross-Dataset** | {cross_m['AUC_mean']:.4f} ± {cross_m['AUC_std']:.4f} |",
                f"| **FMR com `τ*` da CASIA** | {cross_m['FMR_mean'] * 100:.2f}% ± {cross_m['FMR_std'] * 100:.2f}% |",
                f"| **FNMR com `τ*` da CASIA** | {cross_m['FNMR_mean'] * 100:.2f}% ± {cross_m['FNMR_std'] * 100:.2f}% |",
                f"| **TAR@FAR=1%** | {cross_m['TAR_at_FAR_1pct_mean'] * 100:.2f}% ± {cross_m['TAR_at_FAR_1pct_std'] * 100:.2f}% |",
            ]
        )

    rob_file = results_dir / "robustness_metrics.json"
    if rob_file.exists():
        with open(rob_file, "r", encoding="utf-8") as f:
            rob_m = json.load(f)
        lines.extend(
            [
                "\n## 4. Experimento D — Robustez em 4 Níveis (`CASIA-Iris-V3 Interval`)\n",
                "| Perturbação | Nível | Parâmetro | EER (%) | AUC | FMR @ `τ*` (%) | FNMR @ `τ*` (%) | TAR@FAR=1% (%) |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for lvl, d in rob_m["gaussian_blur"].items():
            lines.append(
                f"| **Blur Gaussiano** | {lvl.capitalize()} | $\\sigma={d['sigma']}$ | {d['EER'] * 100:.2f}% | {d['AUC']:.4f} | {d['FMR'] * 100:.2f}% | {d['FNMR'] * 100:.2f}% | {d['TAR_at_FAR_1pct'] * 100:.2f}% |"
            )
        for lvl, d in rob_m["biometric_occlusion"].items():
            lines.append(
                f"| **Oclusão Biométrica** | {lvl.capitalize()} | {int(d['occlusion_ratio'] * 100)}% | {d['EER'] * 100:.2f}% | {d['AUC']:.4f} | {d['FMR'] * 100:.2f}% | {d['FNMR'] * 100:.2f}% | {d['TAR_at_FAR_1pct'] * 100:.2f}% |"
            )

    abl_file = results_dir / "ablation_comparison.json"
    if abl_file.exists():
        with open(abl_file, "r", encoding="utf-8") as f:
            abl_m = json.load(f)
        c_rs = abl_m["com_rubber_sheet"]
        s_rs = abl_m["sem_rubber_sheet"]
        lines.extend(
            [
                "\n## 5. Estudo de Ablação — Com vs. Sem Rubber Sheet de Daugman\n",
                "| Configuração | `use_rubber_sheet` | EER (%) | AUC | FMR @ `τ*` (%) | FNMR @ `τ*` (%) | TAR@FAR=1% (%) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
                f"| **Com Rubber Sheet ($64\\times 512 \\rightarrow 224\\times 224$)** | `true` | {c_rs['EER_mean'] * 100:.2f}% ± {c_rs['EER_std'] * 100:.2f}% | {c_rs['AUC_mean']:.4f} ± {c_rs['AUC_std']:.4f} | {c_rs['FMR_mean'] * 100:.2f}% | {c_rs['FNMR_mean'] * 100:.2f}% | {c_rs['TAR_at_FAR_1pct_mean'] * 100:.2f}% |",
                f"| **Sem Rubber Sheet (ROI 2D Direta $224\\times 224$)** | `false` | {s_rs['EER_mean'] * 100:.2f}% | {s_rs['AUC_mean']:.4f} | {s_rs['FMR_mean'] * 100:.2f}% | {s_rs['FNMR_mean'] * 100:.2f}% | {s_rs['TAR_at_FAR_1pct_mean'] * 100:.2f}% |",
            ]
        )

    out_md = results_dir / "final_summary_table.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Tabela padronizada em Markdown gerada em '{out_md.as_posix()}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo de Verificação Biométrica de Íris 1:1 (ResNet-50)."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho para config.yaml.")
    parser.add_argument("--skip-train", action="store_true", help="Pula o treinamento caso checkpoints já existam.")
    args = parser.parse_args()

    t_start = time.perf_counter()
    cfg = load_config(args.config)
    seeds = [int(s) for s in cfg.get("seeds", [42, 123, 2026])]

    # Etapa 1: Verificação das bases oficiais organizadas localmente em data/
    logger.info(">>> [ETAPA 1/8] Verificando organização local das bases oficiais em data/...")
    verify_local_datasets(cfg)

    # Etapa 2: Geração dos protocolos (seed = 42, sem Identity Leakage, Olho L != R)
    logger.info(">>> [ETAPA 2/8] Geração dos protocolos experimentais (seed=42, Olho L != R)...")
    protocols_main()

    # Etapa 3: Exemplos visuais reais do pipeline para o artigo
    logger.info(">>> [ETAPA 3/8] Gerando figura de exemplos reais das etapas do pipeline (results/pipeline_examples.png)...")
    save_pipeline_examples(config_path=args.config, output_path="results/pipeline_examples.png")

    # Etapa 4: Treinamento das 3 seeds na CASIA-Iris-Interval
    if not args.skip_train:
        logger.info(f">>> [ETAPA 4/8] Treinando ResNet-50 com CosFace na CASIA-Iris-Interval para seeds={seeds}...")
        run_training(config_path=args.config, seeds_override=seeds, use_rubber_sheet=True)

    # Etapa 5: Extração de embeddings L2-normalizados para todas as seeds
    logger.info(f">>> [ETAPA 5/8] Extraindo embeddings L2-normalizados e medindo latência de inferência para seeds={seeds}...")
    for s in seeds:
        run_extraction(config_path=args.config, seed=s, use_rubber_sheet=True)

    # Etapa 6: Avaliação na CASIA (Experimento A), Cross-Dataset na IITD (Experimento C) e Análise de Erros
    logger.info(">>> [ETAPA 6/8] Executando Experimento A (CASIA), Experimento C (Cross-Dataset IITD) e Análise de Erros...")
    run_evaluation(config_path=args.config, use_rubber_sheet=True)

    # Etapa 7: Experimento D — Avaliação de Robustez em 4 Níveis (Blur Gaussiano e Oclusão Biométrica)
    logger.info(">>> [ETAPA 7/8] Executando Experimento D — Teste de Robustez em 4 Níveis (Blur e Oclusão)...")
    run_robustness_experiment(config_path=args.config)

    # Etapa 8: Estudo de Ablação (Com vs. Sem Rubber Sheet)
    logger.info(">>> [ETAPA 8/8] Executando Estudo de Ablação (Com Rubber Sheet vs. Sem Rubber Sheet)...")
    run_ablation_study(config_path=args.config, train_if_missing=not args.skip_train)

    # Gera tabela padronizada final em Markdown
    generate_markdown_summary_table(Path("results"))

    elapsed_min = (time.perf_counter() - t_start) / 60.0
    logger.info("=" * 75)
    logger.info(f"PIPELINE COMPLETO CONCLUÍDO COM SUCESSO EM {elapsed_min:.2f} MINUTOS!")
    logger.info("Todos os artefatos, métricas, scores e figuras foram salvos em 'results/'.")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
