# Verificação Biométrica de Íris 1:1 com ResNet-50 e CosFace (Grupo 5)

Implementação completa, reprodutível e cientificamente auditável de um sistema de **Verificação Biométrica 1:1 de Íris** baseado na arquitetura **ResNet-50** com aprendizado métrico angular (**CosFace**), desenvolvida para o Projeto Final da disciplina de **Biometria** (UTFPR — Câmpus Ponta Grossa).

---

## 1. Visão Geral e Especificações Técnicas

| Item | Especificação |
| :--- | :--- |
| **Grupo / Modalidade** | **Grupo 5 — Íris** (Verificação 1:1 por *Embeddings*) |
| **Regra de Identidade Biométrica** | Olho Esquerdo (`L`) ≠ Olho Direito (`R`) do mesmo sujeito são tratados como **identidades biométricas estritamente distintas** |
| **Base Principal (Treino / Val / Teste)** | `CASIA-Iris-V3 Interval` (`data/CASIA-Iris-Interval`) |
| **Segunda Base (Cross-Dataset)** | `IIT Delhi Iris Database — IITD` (`data/IITD`) |
| **Arquitetura CNN** | `ResNet-50` (`torchvision.models.resnet50`) |
| **Origem dos Pesos / Pré-treinamento** | `ResNet50_Weights.IMAGENET1K_V2` — Pré-treinada no `ImageNet-1K` |
| **Camadas Congeladas (`requires_grad=False`)** | `conv1`, `bn1`, `layer1`, `layer2` (*BatchNorms* mantidos em `.eval()`) |
| **Camadas Treinadas (`requires_grad=True`)** | `layer3` (1024 canais), `layer4` (2048 canais) e `projection_head` |
| **Cabeça de Projeção (`ProjectionHead`)** | `Linear(2048→512, bias=False) → BN1d(512) → PReLU → Dropout(0.2) → Linear(512→256, bias=False) → BN1d(256)` |
| **Dimensão do Embedding** | `256` (normalizado na esfera unitária L2: `e_norm = e / ||e||_2`) |
| **Nº de Parâmetros do Modelo** | Total: `24,690,240` (~24.69M) \| Treináveis: `23,245,312` (~23.25M) \| Tamanho: `~94.5 MB` |
| **Função de Perda** | `CosFace` (*Large Margin Cosine Loss*, `s = 30.0`, `m = 0.35`) \| Opção secundária: `BatchHardTripletLoss` |
| **Otimizador e Treinamento** | `AdamW` (`lr_backbone=1e-4`, `lr_head=3e-4`, `weight_decay=1e-4`), `CosineAnnealingLR` (25 épocas) e *Mixed Precision* (`torch.amp.autocast('cuda')`) |
| **Métrica de Matching** | **Similaridade de Cosseno** entre *embeddings* normalizados em L2: `score(A, B) = dot(emb_A, emb_B)` |
| **Regra de Definição do Threshold (`τ*`)** | Definido **exclusivamente** no conjunto de Validação (`protocols/val_pairs.csv`) no ponto onde `FMR(τ*) = FNMR(τ*)` (EER de validação) e **congelado** antes da avaliação em Teste e Cross-Dataset |
| **Sementes de Reprodutibilidade** | Protocolo e divisão de identidades: `seed = 42` \| Repetição experimental de treino: `seeds = [42, 123, 2026]` |

---

## 2. Obtenção Legal e Organização Local das Bases de Dados

Em estrita conformidade com as regras do projeto (**Seção 22** e **Regra 15 da Seção 29**), **os datasets originais não são redistribuídos neste repositório**.

### 2.1 Como Obter Legalmente as Bases Oficiais
1. **CASIA-Iris-V3 Interval (Base Principal — 249 sujeitos, 2.639 imagens `.jpg`, `320 × 280` px):**
   - Desenvolvida originalmente pelo *Center for Biometrics and Security Research (CBSR) / Institute of Automation, Chinese Academy of Sciences (CASIA)*. Como o portal legado da CASIA (`biometrics.idealtest.org`) encontra-se frequentemente indisponível fora da rede acadêmica chinesa, o pacote original integral (`CASIA-Iris-Interval.zip`, contendo as 2.639 imagens `.jpg` oficiais) pode ser baixado diretamente pelos repositórios acadêmicos ativos abaixo:
     - **Hugging Face Datasets (Navegação e Download Direto):** https://huggingface.co/datasets/chitradrishti/CASIA-IRIS/tree/main
     - **Link Direto do Arquivo (`CASIA-Iris-Interval.zip`):** https://huggingface.co/datasets/chitradrishti/CASIA-IRIS/resolve/main/CASIA-Iris-Interval.zip
     - **Kaggle Datasets (Repositório alternativo):** https://www.kaggle.com/datasets/swoyam2609/casia-iris-interval
2. **IIT Delhi Iris Database — IITD v1.0 (Segunda Base / Cross-Dataset — 218 sujeitos, 2.180 imagens `.bmp`, `320 × 240` px):**
   - Solicite acesso oficial junto ao *Biometrics Research Laboratory — Indian Institute of Technology Delhi (IIT Delhi)* coordenado pelo Prof. Ajay Kumar: https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Iris.htm
   - Referência obrigatória da base: *Ajay Kumar and Arun Passi, "Comparison and combination of iris matchers for reliable personal identification," Pattern Recognition, vol. 43, no. 3, pp. 1016-1026, March 2010.*

### 2.2 Como Organizar Localmente em `data/`
Após obter as duas bases oficiais, extraia os arquivos dentro da pasta `data/` na raiz do projeto seguindo exatamente a estrutura abaixo:

```text
data/
├── CASIA-Iris-Interval/
│   ├── 001/
│   │   ├── L/S1001L01.jpg ...
│   │   └── R/S1001R01.jpg ...
│   └── ... (001 a 249 — 2.639 imagens .jpg)
└── IITD/
    ├── 001/
    │   ├── 01_L.bmp ... 05_L.bmp
    │   └── 06_R.bmp ... 10_R.bmp
    └── ... (001 a 218 — 2.180 imagens .bmp)
```

---

## 3. Pipeline de Pré-processamento e Estudo de Ablação

O módulo [`preprocess.py`](preprocess.py) implementa todas as etapas obrigatórias para a modalidade Íris:

```text
Dado Bruto (NIR) → Controle de Qualidade (FTA) → Inpainting de Reflexos Especulares → Segmentação Pupila/Limbo → Rubber Sheet (64×512) → CLAHE → Resize (224×224) → Normalização ImageNet
```

1. **Controle de Qualidade e FTA (*Failure to Acquire*):** Verifica integridade de leitura e nitidez mínima via variância do operador Laplaciano (`Var(Laplaciano) > 3.0`). Amostras corrompidas ou reprovadas são registradas em `results/fta_report.json`.
2. **Remoção de Reflexos Especulares:** Pixels saturados (`> 200`) são detectados por máscara binária dilatada e reconstruídos com `cv2.inpaint` (`INPAINT_TELEA`).
3. **Localização da Pupila `(xp, yp, rp)` e do Limbo `(xi, yi, ri)`:**
   - **Pupila:** Suavização Gaussiana (`7×7`), limiarização adaptativa/Otsu invertida na região central, abertura/fechamento morfológico e seleção de contorno por circularidade ou `cv2.HoughCircles`.
   - **Limbo/Esclera:** Gradiente/Canny + `cv2.HoughCircles` restrito ao intervalo radial `[1.85 * rp, 3.2 * rp]` e centro próximo a `(xp, yp)`. Em amostras ruidosas, aciona **fallback geométrico concêntrico** em `(xp, yp)` com `ri = 2.2 * rp`.
4. **Modelo *Rubber Sheet* de Daugman e Normalização:**
   - Desenrola o anel iridiano para coordenadas polares adimensionais `(r, θ)` de dimensão `64 × 512` via interpolação bilinear (`cv2.remap`).
   - Aplica **CLAHE** (`clipLimit=2.0`, `tileGridSize=(8,8)`), redimensiona para `224 × 224`, replica para 3 canais e normaliza com `mean=[0.485, 0.456, 0.406]` e `std=[0.229, 0.224, 0.225]`.
5. **Modo de Estudo de Ablação (`use_rubber_sheet: False`):**
   - Recorta diretamente o *bounding box* quadrado da íris `(xi ± ri, yi ± ri)` sem mapeamento polar, aplica máscara circular externa + CLAHE e redimensiona para `224 × 224`.

---

## 4. Estrutura Completa do Repositório

```text
projeto/
├── README.md                  # Documentação completa do projeto e instruções de reprodução
├── .gitignore                 # Ignora data/, checkpoints/, __pycache__/, *.pth, *.pt, *.onnx, *.npz
├── requirements.txt           # Dependências Python do projeto
├── config.yaml                # Hiperparâmetros centralizados do pipeline, modelo e protocolo
├── generate_protocols.py      # Split por identidade (70/15/15, seed=42, Olho L!=R) e geração de pares
├── preprocess.py              # Qualidade (FTA), segmentação pupila/limbo, Rubber Sheet e CLAHE
├── model.py                   # ResNet50IrisEmbedder, ProjectionHead, CosFaceLoss e TripletLoss
├── train.py                   # Treinamento multi-seed (42, 123, 2026) com validação por EER
├── extract_embeddings.py      # Extração de embeddings L2 e benchmark de tempo de inferência/hardware
├── evaluate.py                # Experimentos A (CASIA), C (Cross-Dataset IITD) e Análise de Erros
├── robustness.py              # Experimento D (Robustez em 4 níveis) e Estudo de Ablação
├── run_all.py                 # Orquestrador que executa todo o pipeline de ponta a ponta
├── protocols/                 # Arquivos CSV de protocolo gerados com seed=42
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   ├── val_pairs.csv
│   ├── genuine_pairs.csv
│   ├── impostor_pairs.csv
│   ├── test_pairs.csv
│   └── cross_dataset_test_pairs.csv
└── results/                   # Métricas, scores e figuras geradas para o artigo científico
    ├── scores.csv
    ├── metrics.json
    ├── roc.png
    ├── distributions.png
    ├── pipeline_examples.png
    ├── error_analysis.csv
    ├── error_analysis.png
    ├── robustness_metrics.json
    ├── robustness_curve.png
    ├── ablation_comparison.json
    ├── ablation_plot.png
    ├── final_summary_table.md
    └── cross_dataset_iitd/
        ├── scores.csv
        ├── metrics.json
        ├── roc.png
        └── distributions.png
```

---

## 5. Ambiente, Dependências e Reprodutibilidade

### 5.1 Versões Recomendadas
- **Python:** `>= 3.10`
- **Principais Bibliotecas (`requirements.txt`):** `torch>=2.1`, `torchvision>=0.16`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `pyyaml`, `tqdm`.
- **Hardware Utilizado:** `GPU: NVIDIA GeForce RTX 3050 | CPU: Intel Core (6 logical cores)` (detectado e registrado automaticamente em `results/metrics.json`).

Instale os pacotes com:
```bash
pip install -r requirements.txt
```

### 5.2 Execução Completa com Um Único Comando
Após organizar as bases oficiais em `data/CASIA-Iris-Interval` e `data/IITD`, execute todo o fluxo experimental (verificação das bases → protocolos → exemplos do pipeline → treino nas 3 seeds → extração → Experimento A → Experimento C Cross-Dataset → Experimento D Robustez → Estudo de Ablação):

```bash
python run_all.py
```

### 5.3 Comandos Individuais Etapa por Etapa

1. **Gerar os protocolos de treino, validação, teste e pares (`seed = 42`, Olho `L` ≠ `R`):**
   ```bash
   python generate_protocols.py
   ```

2. **Gerar a figura de exemplos reais do pipeline (`results/pipeline_examples.png`):**
   ```bash
   python preprocess.py
   ```

3. **Treinar a ResNet-50 nas 3 seeds (`42`, `123`, `2026`) na base principal:**
   ```bash
   python train.py
   ```

4. **Extrair embeddings L2-normalizados e medir tempo médio de inferência por amostra:**
   ```bash
   python extract_embeddings.py --all-seeds
   ```

5. **Executar Experimento A (CASIA), Experimento C (Cross-Dataset IITD) e Análise de Erros:**
   ```bash
   python evaluate.py
   ```

6. **Executar Experimento D (Robustez em 4 Níveis: Blur e Oclusão) e Estudo de Ablação (Com vs. Sem Rubber Sheet):**
   ```bash
   python robustness.py
   ```
