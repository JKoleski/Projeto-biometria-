# Consolidação Completa dos Resultados Experimentais — Grupo 5 (Íris / ResNet-50) e Comparativo com Grupo 6 (MobileNetV3-Small)

Documento consolidado com todas as tabelas exigidas no documento de requisitos (`Projeto_Final_Disciplina_Biometria.pdf`), utilizando **exatamente os mesmos `4.156` pares de teste da base IITD (`756` genuínos e `3.400` impostores)** compartilhados com o **Grupo 6** (conforme a **Seção 11** e **Regra 8**).

---

## 1. Tabela Padronizada e Experimento A — Base Principal (`CASIA-Iris-V3 Interval`)
*(Média ± Desvio-Padrão das 3 seeds: `42`, `123` e `2026`)*

| Informação | Resultado |
| :--- | :--- |
| **Modalidade** | Íris (Olho Esquerdo `L` ≠ Olho Direito `R`) |
| **Dataset Principal** | `CASIA-Iris-V3 Interval` (249 sujeitos, 2.639 imagens `.jpg`, `320 × 280` px) |
| **Dataset Secundário (Cross-Dataset)** | `IIT Delhi Iris v1.0` (218 sujeitos, 2.180 imagens `.bmp`, `320 × 240` px — `4.156` pares sincronizados com Grupo 6) |
| **CNN** | `ResNet-50` |
| **Pré-treinamento / Origem dos Pesos** | `ImageNet-1K` (`ResNet50_Weights.IMAGENET1K_V2`) |
| **Camadas Congeladas** | `conv1, bn1, layer1, layer2` |
| **Camadas Treinadas** | `layer3, layer4, projection_head` |
| **Função de Perda** | `CosFace` (`s=30.0`, `m=0.35`) |
| **Nº de Parâmetros (Total / Treináveis)** | `24.689.217` (24,69M) / `23.244.289` (23,24M) |
| **Dimensão do Embedding** | `256` (normalizado na esfera unitária L2) |
| **Matching** | Similaridade de Cosseno (`s = dot(emb_A, emb_B)`) |
| **Threshold Congelado na Validação (`τ*`)** | `0,2038` (Seed 42) \| Média das 3 seeds: `0,1993 ± 0,0052` |
| **EER (Média ± DP)** | **12,31% ± 1,25%** (Melhor seed: **11,22%** na seed `2026`) |
| **AUC (Média ± DP)** | **0,9480 ± 0,0091** (Melhor seed: **0,9584** na seed `2026`) |
| **FMR / FAR (`τ*` congelado)** | **13,86% ± 0,67%** |
| **FNMR / FRR (`τ*` congelado)** | **11,11% ± 1,71%** |
| **TAR@FAR=1% (Média ± DP)** | **67,50% ± 3,82%** (Melhor seed: **71,51%** na seed `2026`) |
| **FTA (*Failure to Acquire*)** | **0,00%** |
| **Tempo Médio de Inferência** | **2,368 ms / amostra** (`1,74 ms` em regime *warm* na GPU) |
| **Tamanho do Modelo** | **94,50 MB** |
| **Hardware Utilizado** | `GPU: NVIDIA GeForce RTX 3050 | CPU: Intel Core (6 logical cores)` |

---

## 2. Experimento B — Matriz Completa $2\text{ CNNs} \times 2\text{ Datasets}$ (Grupo 5 vs. Grupo 6)

| Arquitetura CNN (Grupo) | Base A: `CASIA-Iris-V3 Interval` (EER / AUC) | Base B: `IIT Delhi Iris v1.0` (`4.156` pares idênticos) | Nº de Parâmetros | Tamanho (MB) | Inferência (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50 (Grupo 5 — Nosso)** | **EER: 12,31% ± 1,25%**<br>**AUC: 0,9480 ± 0,0091**<br>*(Base Principal)* | **EER: 18,03% ± 0,80%**<br>**AUC: 0,8973 ± 0,0078**<br>*(Cross-Dataset)* | 24.689.217 (24,69M) | 94,50 MB | 2,37 ms (GPU) |
| **MobileNetV3-Small (Grupo 6)** | **EER: 25,95% ± 1,01%**<br>**AUC: 0,8150 ± 0,0124**<br>*(Cross-Dataset)* | **EER: 10,74% ± 0,92%**<br>**AUC: 0,9478 ± 0,0044**<br>*(Base Principal)* | 1.289.376 (1,29M) | 5,05 MB | 15,98 ms (CPU) |

---

## 3. Experimento C — Cross-Dataset na `IITD` Oficial (`4.156` Pares Sincronizados com Grupo 6)
*(756 pares genuínos + 3.400 pares impostores do conjunto de teste da IITD do Grupo 6)*

| Seed | Threshold Congelado da CASIA (`τ*`) | EER Cross-Dataset (%) | AUC Cross-Dataset | FMR com `τ*` da CASIA (%) | FNMR com `τ*` da CASIA (%) | TAR@FAR=1% (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 0,2038 | 18,78% | 0,8909 | 84,12% | 1,19% | 53,97% |
| **123** | 0,1920 | **16,93%** | **0,9082** | 87,53% | **0,93%** | 54,76% |
| **2026** | 0,2022 | 18,39% | 0,8927 | 90,09% | 1,19% | **57,94%** |
| **Média ± DP** | **0,1993 ± 0,0052** | **18,03% ± 0,80%** | **0,8973 ± 0,0078** | **87,25% ± 2,45%** | **1,10% ± 0,12%** | **55,56% ± 1,71%** |

---

## 4. Experimento D — Robustez em 4 Níveis (`CASIA-Iris-V3 Interval`)

| Perturbação | Nível | Parâmetro | EER (%) | AUC | FMR @ `τ*` (%) | FNMR @ `τ*` (%) | TAR@FAR=1% (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Blur Gaussiano** | Original | `σ = 0,0` | 14,07% | 0,9362 | 14,67% | 13,53% | 62,37% |
| **Blur Gaussiano** | Baixa | `σ = 1,5` | 19,27% | 0,8954 | 75,87% | 1,70% | 41,85% |
| **Blur Gaussiano** | Média | `σ = 3,0` | 32,35% | 0,7546 | 99,91% | 0,00% | 16,40% |
| **Blur Gaussiano** | Alta | `σ = 5,0` | 35,04% | 0,7250 | 100,00% | 0,00% | 14,07% |
| **Oclusão Biométrica** | Original | 0% | 14,07% | 0,9362 | 14,67% | 13,53% | 62,37% |
| **Oclusão Biométrica** | Baixa | 15% | 24,37% | 0,8501 | 40,34% | 13,35% | 35,48% |
| **Oclusão Biométrica** | Média | 30% | 32,93% | 0,7295 | 87,46% | 3,58% | 12,46% |
| **Oclusão Biométrica** | Alta | 45% | 40,82% | 0,6511 | 99,57% | 0,36% | 11,83% |

---

## 5. Estudo de Ablação — Com vs. Sem *Rubber Sheet* de Daugman

| Configuração | `use_rubber_sheet` | EER (%) | AUC | FMR @ `τ*` (%) | FNMR @ `τ*` (%) | TAR@FAR=1% (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Com Rubber Sheet (`64×512 → 224×224`)** | `true` | 12,31% ± 1,25% | 0,9480 ± 0,0091 | 13,86% | 11,11% | 67,50% |
| **Sem Rubber Sheet (ROI 2D Direta `224×224`)** | `false` | **11,02%** | **0,9601** | **12,97%** | **9,59%** | **68,73%** |
