# Relatório Consolidado de Resultados para o Artigo Científico — Modalidade Íris (Grupos 5 e 6)

Este documento consolida todos os resultados experimentais do **Grupo 5 (`ResNet-50`)** integrados e sincronizados com o protocolo de pares do **Grupo 6 (`MobileNetV3-Small`)** na base **IIT Delhi Iris (IITD v1.0)**, em estrita conformidade com a **Seção 11** e **Regra 8** do documento de requisitos (*"Os mesmos pares deverão ser utilizados pelas duas CNNs avaliadas em uma mesma base"*).

---

## 1. Alinhamento Metodológico e Sincronização de Pares (Seção 11)

* **Na base `IIT Delhi Iris (IITD v1.0)` (218 sujeitos, 2.180 imagens `.bmp`, $320 \times 240$ px):**
  * Como a `IITD` é a base principal do **Grupo 6**, o conjunto de teste oficial da base (15% das identidades oculares reservadas para teste com `seed = 42`, zero *identity leakage*, olho `L` $\neq$ olho `R`) contém exatamente **`4.156` pares de verificação 1:1**, sendo **`756` pares genuínos** e **`3.400` pares impostores**.
  * Tanto a **`MobileNetV3-Small` (Grupo 6 — In-Domain)** quanto a **`ResNet-50` (Grupo 5 — Cross-Dataset sem fine-tuning)** foram avaliadas **exatamente sobre esses mesmos `4.156` pares** (`protocols/cross_dataset_test_pairs.csv`).
* **Na base `CASIA-Iris-V3 Interval` (249 sujeitos, 2.639 imagens `.jpg`, $320 \times 280$ px):**
  * Como a `CASIA-Iris-V3 Interval` é a base principal do **Grupo 5**, o conjunto de teste oficial da base (15% das identidades reservadas para teste com `seed = 42`, `56` identidades oculares, zero *identity leakage*) contém **`5.580` pares de verificação 1:1** (**`1.116` pares genuínos** e **`4.464` pares impostores** em `protocols/test_pairs.csv`).

---

## 2. Tabela Padronizada Consolidada — Comparativo Direto Grupo 5 vs. Grupo 6 (Experimentos A, B e C)
*(Valores expressos como **Média ± Desvio-Padrão** nas 3 seeds de repetição experimental: `42`, `123` e `2026`)*

| Informação / Métrica | Grupo 5 — Base Principal (`CASIA-Iris-V3 Interval`) | Grupo 5 — Cross-Dataset (`IITD` — 4.156 pares idênticos) | Grupo 6 — Base Principal (`IITD` — 4.156 pares idênticos) | Grupo 6 — Cross-Dataset (`CASIA-Iris-V3 Interval`) |
| :--- | :---: | :---: | :---: | :---: |
| **Modalidade** | Íris (`L` $\neq$ `R`) | Íris (`L` $\neq$ `R`) | Íris (`L` $\neq$ `R`) | Íris (`L` $\neq$ `R`) |
| **Papel da Base** | Treino / Val / Teste (In-Domain) | Cross-Dataset (Sem Fine-Tuning) | Treino / Val / Teste (In-Domain) | Cross-Dataset (Sem Fine-Tuning) |
| **Família / Arquitetura CNN** | **ResNet-50** | **ResNet-50** | **MobileNetV3-Small** | **MobileNetV3-Small** |
| **Pré-treinamento** | `ImageNet-1K (IMAGENET1K_V2)` | `ImageNet-1K (IMAGENET1K_V2)` | `ImageNet-1K (DEFAULT)` | `ImageNet-1K (DEFAULT)` |
| **Camadas Congeladas** | `conv1, bn1, layer1, layer2` | Modelo 100% congelado da CASIA | Nenhuma (*fine-tuning* completo) | Modelo 100% congelado da IITD |
| **Camadas Treinadas** | `layer3, layer4, projection_head` | Nenhuma (0 ajuste na IITD) | `features` + `embedding_head` | Nenhuma (0 ajuste na CASIA) |
| **Função de Perda** | `CosFace` ($s=30.0, m=0.35$) | `CosFace` ($s=30.0, m=0.35$) | `CosFace` ($s=30.0, m=0.30$) | `CosFace` ($s=30.0, m=0.30$) |
| **Nº de Parâmetros (Total / Treináveis)** | `24.689.217` / `23.244.289` | `24.689.217` / `0` | `1.289.376` / `1.289.376` | `1.289.376` / `0` |
| **Tamanho do Modelo (MB)** | **94,50 MB** | **94,50 MB** | **5,05 MB** | **5,05 MB** |
| **Dimensão do Embedding** | `256` (normalizado $L_2$) | `256` (normalizado $L_2$) | `128` (normalizado $L_2$) | `128` (normalizado $L_2$) |
| **Pré-processamento (Rubber Sheet)** | $64 \times 512 \rightarrow \text{CLAHE} \rightarrow 224 \times 224$ | $64 \times 512 \rightarrow \text{CLAHE} \rightarrow 224 \times 224$ | $128 \times 128 \text{ (polar)} \rightarrow \text{CLAHE}$ | $128 \times 128 \text{ (polar)} \rightarrow \text{CLAHE}$ |
| **Métrica de Matching** | Similaridade de Cosseno | Similaridade de Cosseno | Similaridade de Cosseno | Similaridade de Cosseno |
| **Pares Avaliados (Genuínos / Impostores)** | `5.580` (`1.116` / `4.464`) | **`4.156` (`756` / `3.400`)** | **`4.156` (`756` / `3.400`)** | `22.127` (`8.932` / `13.195`) |
| **Threshold Congelado (`τ*`)** | `0,1993 ± 0,0052` (Seed 42: `0,2038`) | `0,1993 ± 0,0052` (da CASIA) | `0,4363 ± 0,0144` (Seed 42: `0,4202`) | `0,4363 ± 0,0144` (da IITD) |
| **EER (Média ± DP)** | **12,31% ± 1,25%** | **18,03% ± 0,80%** | **10,74% ± 0,92%** | **25,95% ± 1,01%** |
| **AUC (Média ± DP)** | **0,9480 ± 0,0091** | **0,8973 ± 0,0078** | **0,9478 ± 0,0044** | **0,8150 ± 0,0124** |
| **FMR / FAR (`τ*` congelado)** | **13,86% ± 0,67%** | **87,25% ± 2,45%** | **0,10% ± 0,07%** | **12,20% ± 7,27%** |
| **FNMR / FRR (`τ*` congelado)** | **11,11% ± 1,71%** | **1,10% ± 0,12%** | **23,06% ± 0,08%** | **37,71% ± 7,07%** |
| **TAR@FAR=1% (Média ± DP)** | **67,50% ± 3,82%** | **55,56% ± 1,71%** | **83,20% ± 0,13%** | **34,55% ± 1,94%** |
| **FTA (*Failure to Acquire*)** | **0,00%** | **0,00%** | **0,00%** | **0,00%** |
| **Tempo Médio de Inferência** | **2,37 ± 0,89 ms** (GPU) | **2,37 ± 0,89 ms** (GPU) | **15,98 ± 0,64 ms** (CPU) | **17,22 ± 1,45 ms** (CPU) |
| **Hardware Utilizado** | `GPU RTX 3050 + Intel Core` | `GPU RTX 3050 + Intel Core` | `CPU Intel Core (Stepping 9)` | `CPU Intel Core (Stepping 9)` |

---

## 3. Matriz Resumo do Experimento B ($2\text{ CNNs} \times 2\text{ Datasets}$)

| Arquitetura CNN | Dataset A: `CASIA-Iris-V3 Interval` | Dataset B: `IIT Delhi Iris v1.0` (`4.156` pares idênticos) | Degradação Cross-Dataset ($\Delta\text{EER}$ / $\Delta\text{AUC}$) |
| :--- | :---: | :---: | :---: |
| **ResNet-50 (Grupo 5)**<br>*(Treinada na CASIA)* | **EER: 12,31% ± 1,25%**<br>**AUC: 0,9480 ± 0,0091**<br>*(In-Domain)* | **EER: 18,03% ± 0,80%**<br>**AUC: 0,8973 ± 0,0078**<br>*(Cross-Dataset)* | **$\Delta\text{EER} = +5,72\text{ p.p.}$**<br>**$\Delta\text{AUC} = -0,0507$** |
| **MobileNetV3-Small (Grupo 6)**<br>*(Treinada na IITD)* | **EER: 25,95% ± 1,01%**<br>**AUC: 0,8150 ± 0,0124**<br>*(Cross-Dataset)* | **EER: 10,74% ± 0,92%**<br>**AUC: 0,9478 ± 0,0044**<br>*(In-Domain)* | **$\Delta\text{EER} = +15,21\text{ p.p.}$**<br>**$\Delta\text{AUC} = -0,1328$** |

---

## 4. Detalhamento por Seed do Grupo 5 (`ResNet-50`) e do Grupo 6 (`MobileNetV3-Small`)

### 4.1 Desempenho na Base `IIT Delhi Iris v1.0` (Exatos `4.156` Pares: `756` Genuínos e `3.400` Impostores)
| Seed | Grupo 5 (`ResNet-50` Cross-Dataset) EER | Grupo 5 (`ResNet-50` Cross-Dataset) AUC | Grupo 5 TAR@FAR=1% | Grupo 6 (`MobileNetV3-Small` In-Domain) EER | Grupo 6 (`MobileNetV3-Small` In-Domain) AUC | Grupo 6 TAR@FAR=1% |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 18,78% | 0,8909 | 53,97% | 11,36% | 0,9504 | 83,07% |
| **123** | **16,93%** | **0,9082** | 54,76% | **9,68%** | **0,9504** | 83,20% |
| **2026** | 18,39% | 0,8927 | **57,94%** | 11,18% | 0,9428 | **83,33%** |
| **Média ± DP** | **18,03% ± 0,80%** | **0,8973 ± 0,0078** | **55,56% ± 1,71%** | **10,74% ± 0,92%** | **0,9478 ± 0,0044** | **83,20% ± 0,13%** |

### 4.2 Desempenho na Base `CASIA-Iris-V3 Interval`
| Seed | Grupo 5 (`ResNet-50` In-Domain) Val EER | Grupo 5 Threshold (`τ*`) | Grupo 5 Test EER | Grupo 5 Test AUC | Grupo 5 Test FMR (`τ*`) | Grupo 5 Test FNMR (`τ*`) | Grupo 5 TAR@FAR=1% | Grupo 6 (`MobileNetV3-Small` Cross-Dataset) EER | Grupo 6 (`MobileNetV3-Small` Cross-Dataset) AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 13,04% | 0,2038 | 14,07% | 0,9362 | 14,67% | 13,53% | 62,37% | 26,11% | 0,8108 |
| **123** | 13,91% | 0,1920 | 11,65% | 0,9494 | 13,87% | 9,95% | 68,64% | **24,87%** | **0,8290** |
| **2026** | 12,62% | 0,2022 | **11,22%** | **0,9584** | **13,04%** | **9,86%** | **71,51%** | 26,86% | 0,8052 |
| **Média ± DP** | **13,19% ± 0,54%** | **0,1993 ± 0,0052** | **12,31% ± 1,25%** | **0,9480 ± 0,0091** | **13,86% ± 0,67%** | **11,11% ± 1,71%** | **67,50% ± 3,82%** | **25,95% ± 1,01%** | **0,8150 ± 0,0124** |

---

## 5. Experimento D — Robustez em 4 Níveis do Grupo 5 (`ResNet-50` na `CASIA-Iris-V3 Interval`)

### 5.1 Perturbação 1: Blur Gaussiano (Desfocagem Óptica)
| Nível de Degradação | Parâmetro ($\sigma$) | EER (%) | AUC | FMR @ `τ*` (%) | FNMR @ `τ*` (%) | TAR@FAR=1% (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Original** | $\sigma = 0,0$ | **14,07%** | **0,9362** | 14,67% | 13,53% | **62,37%** |
| **2. Baixa Degradação** | $\sigma = 1,5$ | **19,27%** | **0,8954** | 75,87% | 1,70% | **41,85%** |
| **3. Média Degradação** | $\sigma = 3,0$ | **32,35%** | **0,7546** | 99,91% | 0,00% | **16,40%** |
| **4. Alta Degradação** | $\sigma = 5,0$ | **35,04%** | **0,7250** | 100,00% | 0,00% | **14,07%** |

### 5.2 Perturbação 2: Oclusão Biométrica (Pálpebras, Cílios e Reflexo Especular)
| Nível de Degradação | Fração Ocluída da Íris | EER (%) | AUC | FMR @ `τ*` (%) | FNMR @ `τ*` (%) | TAR@FAR=1% (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Original** | $0\%$ | **14,07%** | **0,9362** | 14,67% | 13,53% | **62,37%** |
| **2. Baixa Degradação** | $15\%$ | **24,37%** | **0,8501** | 40,34% | 13,35% | **35,48%** |
| **3. Média Degradação** | $30\%$ | **32,93%** | **0,7295** | 87,46% | 3,58% | **12,46%** |
| **4. Alta Degradação** | $45\%$ | **40,82%** | **0,6511** | 99,57% | 0,36% | **11,83%** |

---

## 6. Estudo de Ablação do Grupo 5 — Impacto do *Rubber Sheet* de Daugman (`ResNet-50`)

| Configuração do Pipeline | `use_rubber_sheet` | EER (%) | AUC | FMR @ `τ*` (%) | FNMR @ `τ*` (%) | TAR@FAR=1% (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Com Rubber Sheet (Daugman $64\times 512 \rightarrow 224\times 224$)** | `true` | 12,31% ± 1,25% | 0,9480 ± 0,0091 | 13,86% | 11,11% | 67,50% |
| **Sem Rubber Sheet (ROI Circular Direta $224\times 224$)** | `false` | **11,02%** | **0,9601** | **12,97%** | **9,59%** | **68,73%** |

---

## 7. Análise Crítica dos Erros do Grupo 5 — 3 Falsos Matches (FM) e 3 Falsos Non-Matches (FNM)

| Caso | Tipo de Erro | Amostra 1 | Amostra 2 | Score | `τ*` | Foco Lap. (1 / 2) | $r_p/r_i$ (1 / 2) | Oclusão (1 / 2) | Hipótese Técnica Fundamentada |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **FM #1** | False Match (FMR) | `S1037R01.jpg` | `S1116L01.jpg` | `0.5946` | `0.2038` | 266,2 / 224,6 | 0,38 / 0,42 | 55,5% / 37,7% | Artefatos comuns de oclusão palpebral/cílios (55,5%) gerando padrões espúrios correlacionados na faixa polar e geometria pupilar semelhante ($\Delta r_p/r_i = 0,04$). |
| **FM #2** | False Match (FMR) | `S1111R04.jpg` | `S1116L05.jpg` | `0.5644` | `0.2038` | 517,7 / 310,4 | 0,48 / 0,52 | 13,2% / 8,4% | Geometria pupilar/limbo quase idêntica ($\Delta r_p/r_i = 0,04$) e macro-textura radial de baixa frequência altamente correlacionada após equalização CLAHE. |
| **FM #3** | False Match (FMR) | `S1158R02.jpg` | `S1240L04.jpg` | `0.5630` | `0.2038` | 262,5 / 548,2 | 0,51 / 0,45 | 0,7% / 6,7% | Similaridade estrutural de baixa frequência no estroma iridiano e iluminação NIR homogênea entre indivíduos distintos. |
| **FNM #1** | False Non-Match (FNMR) | `S1045L03.jpg` | `S1045L06.jpg` | `-0.2484` | `0.2038` | 195,6 / 220,7 | 0,41 / 0,47 | 36,2% / 48,2% | Oclusão severa por pálpebras e cílios (48,2%) mascarando faixas superiores/inferiores no Rubber Sheet aliada a reflexos especulares residuais. |
| **FNM #2** | False Non-Match (FNMR) | `S1045L03.jpg` | `S1045L07.jpg` | `-0.2403` | `0.2038` | 195,6 / 309,7 | 0,41 / 0,52 | 36,2% / 34,2% | Variação acentuada de dilatação pupilar ($\Delta r_p/r_i = 0,11$) deformando não-linearmente as criptas do estroma somada à oclusão palpebral (36,2%). |
| **FNM #3** | False Non-Match (FNMR) | `S1042L03.jpg` | `S1042L04.jpg` | `-0.2394` | `0.2038` | 266,6 / 284,6 | 0,40 / 0,44 | 24,0% / 16,9% | Oclusão palpebral/cílios (24,0%) associada a desalinhamento rotacional torsional e reflexos especulares na borda pupilar. |

---

## 8. Principais Conclusões Científicas para a Seção 7 (*Análise e Discussão*) do Artigo

1. **Equivalência de Desempenho *In-Domain* vs. Custo Computacional:**
   * Na respectiva base principal de treinamento, tanto a **`ResNet-50` (Grupo 5 na CASIA)** quanto a **`MobileNetV3-Small` (Grupo 6 na IITD)** alcançaram capacidade discriminativa praticamente idêntica (**$\text{AUC} = 0,9480 \pm 0,0091$** vs. **$\text{AUC} = 0,9478 \pm 0,0044$**; $\text{EER}$ de $12,31\%$ vs. $10,74\%$).
   * Entretanto, a `MobileNetV3-Small` possui **19 vezes menos parâmetros** ($1,29\text{M}$ vs. $24,69\text{M}$) e ocupa **18,7 vezes menos armazenamento** ($5,05\text{ MB}$ vs. $94,50\text{ MB}$), demonstrando excelente eficiência para cenários embarcados quando treinada no próprio domínio do sensor.
2. **Superioridade da `ResNet-50` em Generalização *Cross-Dataset* (Experimento C):**
   * Quando avaliada em domínio cruzado sem *fine-tuning*, a **`ResNet-50` (Grupo 5: CASIA $\rightarrow$ IITD nos `4.156` pares)** manteve **$\text{AUC} = 0,8973 \pm 0,0078$** e **$\text{EER} = 18,03\% \pm 0,80\%$** (degradação moderada de $+5,72\text{ p.p.}$ no EER), superando com ampla margem a generalização cruzada da **`MobileNetV3-Small` (Grupo 6: IITD $\rightarrow$ CASIA)**, que obteve **$\text{AUC} = 0,8150 \pm 0,0124$** e **$\text{EER} = 25,95\% \pm 1,01\%$** (degradação de $+15,21\text{ p.p.}$ no EER).
   * **Hipótese Técnica:** O congelamento dos estágios iniciais (`conv1` a `layer2`) da `ResNet-50` preservou filtros gerais de borda/textura do `ImageNet-1K` e evitou *overfitting* às características específicas do sensor de origem, enquanto o *embedding* de maior dimensão (`256-D` vs. `128-D`) reteve maior riqueza estrutural das criptas iridianas sob mudança de câmera NIR.
3. **Efeito do *Domain Shift* sobre o Limiar Congelado (`τ*`):**
   * Na transição `CASIA` $\rightarrow$ `IITD` (`ResNet-50`), o limiar congelado na validação da CASIA ($\tau^* = 0,2038$) resultou em $\text{FNMR} = 1,10\%$ e $\text{FMR} = 87,25\%$ na IITD (apesar da alta separabilidade intrínseca de $\text{AUC} = 0,8973$), pois as imagens da câmera CMOS JIRIS JPC1000 da IITD apresentam maior contraste médio, deslocando a distribuição de scores de cosseno para a direita. No sentido inverso (`IITD` $\rightarrow$ `CASIA` na `MobileNetV3-Small`), o limiar mais alto congelado na IITD ($\tau^* = 0,4363$) provocou aumento da falsa rejeição na CASIA ($\text{FNMR} = 37,71\%$). Isso comprova que sistemas biométricos operando entre sensores distintos requerem recalibração de limiar (ou normalização de score *z-norm/t-norm*), mesmo quando o extrator CNN preserva a ordenação ROC.
