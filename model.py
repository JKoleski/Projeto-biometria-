"""
Definição da arquitetura ResNet-50 para Verificação Biométrica de Íris 1:1:
1. ResNet50IrisEmbedder: Backbone ResNet-50 (IMAGENET1K_V2) com congelamento configurável
   (conv1, bn1, layer1, layer2 congeladas; layer3, layer4 e ProjectionHead treináveis) e saída L2-normalizada.
2. CosFaceLoss (Large Margin Cosine Loss, s=30.0, m=0.35) e BatchHardTripletLoss (distância de cosseno).
3. get_model_info: Metadados completos de complexidade, camadas congeladas/treinadas, nº de parâmetros e tamanho em MB.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ProjectionHead(nn.Module):
    """
    Cabeça de projeção métrica substituindo a camada fc original (2048 -> 1000) da ResNet-50:
    Linear(2048 -> 512, bias=False) -> BatchNorm1d(512) -> PReLU() -> Dropout(0.2)
    -> Linear(512 -> embedding_dim, bias=False) -> BatchNorm1d(embedding_dim)
    """

    def __init__(self, in_features: int = 2048, embedding_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet50IrisEmbedder(nn.Module):
    """
    Extrator de embeddings biométricos de íris baseado na ResNet-50 pré-treinada no ImageNet-1K.
    Retorna sempre embeddings normalizados na esfera unitária L2.
    """

    BACKBONE_STAGES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]

    def __init__(
        self,
        embedding_dim: int = 256,
        freeze_up_to: str = "layer2",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.freeze_up_to = freeze_up_to
        self.pretrained_weights_name = "IMAGENET1K_V2" if pretrained else "None"
        self.pretrained_dataset_name = "ImageNet-1K" if pretrained else "None"

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Mantém as etapas convolucionais da ResNet-50
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Substitui a camada classificadora fc por ProjectionHead
        self.projection_head = ProjectionHead(
            in_features=backbone.fc.in_features,
            embedding_dim=self.embedding_dim,
            dropout=dropout,
        )

        self.frozen_layers: List[str] = []
        self.trained_layers: List[str] = []
        self._apply_freezing(freeze_up_to)

    def _apply_freezing(self, freeze_up_to: Optional[str]) -> None:
        """
        Congela camadas da ResNet-50 até `freeze_up_to` (inclusive).
        Por padrão ('layer2'): congela conv1, bn1, layer1 e layer2;
        mantém layer3, layer4 e projection_head treináveis.
        """
        if not freeze_up_to or freeze_up_to.lower() == "none":
            self.frozen_layers = []
            self.trained_layers = list(self.BACKBONE_STAGES) + ["projection_head"]
            return

        if freeze_up_to not in self.BACKBONE_STAGES:
            raise ValueError(
                f"freeze_up_to='{freeze_up_to}' inválido. Opções válidas: {self.BACKBONE_STAGES}"
            )

        stop_idx = self.BACKBONE_STAGES.index(freeze_up_to)
        self.frozen_layers = self.BACKBONE_STAGES[: stop_idx + 1]
        self.trained_layers = self.BACKBONE_STAGES[stop_idx + 1 :] + ["projection_head"]

        for stage_name in self.frozen_layers:
            stage_module = getattr(self, stage_name)
            for param in stage_module.parameters():
                param.requires_grad = False
            stage_module.eval()

        for stage_name in self.BACKBONE_STAGES[stop_idx + 1 :]:
            stage_module = getattr(self, stage_name)
            for param in stage_module.parameters():
                param.requires_grad = True

        for param in self.projection_head.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True) -> "ResNet50IrisEmbedder":
        """
        Sobrescreve train() para garantir que as camadas BatchNorm congeladas (bn1, layer1, layer2)
        permaneçam em modo eval() durante o fine-tuning, preservando as estatísticas do ImageNet-1K.
        """
        super().train(mode)
        if mode:
            for stage_name in self.frozen_layers:
                getattr(self, stage_name).eval()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        emb = self.projection_head(features)
        return F.normalize(emb, p=2, dim=1)


class CosFaceLoss(nn.Module):
    """
    Large Margin Cosine Loss (CosFace) para aprendizado métrico profundo.
    Otimiza diretamente a similaridade de cosseno com margem angular aditiva m=0.35 e fator de escala s=30.0.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        s: float = 30.0,
        m: float = 0.35,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.s = float(s)
        self.m = float(m)

        self.weight = nn.Parameter(torch.empty(self.num_classes, self.embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normaliza embeddings e centros de classe em L2
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Similaridade de cosseno cos(theta_j)
        cosine = F.linear(emb_norm, weight_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Subtrai margem m apenas da classe verdadeira y_i: cos(theta_y) - m
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine, dtype=torch.bool)
        one_hot.scatter_(1, labels.view(-1, 1).long(), True)

        logits = self.s * torch.where(one_hot, phi, cosine)
        return F.cross_entropy(logits, labels.long())


class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Loss baseada em distância de cosseno (d_cos = 1 - cos(u, v)).
    Disponível como segunda opção configurável via loss_type: 'triplet' no config.yaml.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        # Matriz de similaridade de cosseno (B, B) e distância de cosseno = 1 - sim
        sim_matrix = torch.matmul(emb_norm, emb_norm.t())
        dist_matrix = (1.0 - sim_matrix).clamp(min=0.0)

        labels = labels.view(-1, 1)
        is_pos = labels.eq(labels.t())
        is_neg = ~is_pos

        # Remove a diagonal principal das comparações positivas
        eye = torch.eye(labels.size(0), dtype=torch.bool, device=embeddings.device)
        is_pos = is_pos & ~eye

        # Hardest positive: máxima distância entre amostras da mesma identidade no batch
        pos_dists = dist_matrix.clone()
        pos_dists[~is_pos] = -1.0
        hardest_pos, _ = pos_dists.max(dim=1)

        # Hardest negative: mínima distância entre amostras de identidades diferentes no batch
        neg_dists = dist_matrix.clone()
        neg_dists[~is_neg] = 1e6
        hardest_neg, _ = neg_dists.min(dim=1)

        valid_anchors = hardest_pos >= 0.0
        if not torch.any(valid_anchors):
            return embeddings.sum() * 0.0

        loss = F.relu(hardest_pos[valid_anchors] - hardest_neg[valid_anchors] + self.margin)
        return loss.mean()


def get_model_info(
    model: ResNet50IrisEmbedder,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Union[str, int, float, List[str]]]:
    """
    3. Metadados de Complexidade:
       Retorna dicionário padronizado com arquitetura, origem dos pesos, dataset de pré-treinamento,
       camadas congeladas, camadas treinadas, dimensão do embedding, total de parâmetros,
       parâmetros treináveis e tamanho do modelo em MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        size_bytes = Path(checkpoint_path).stat().st_size
        model_size_mb = round(size_bytes / (1024.0 * 1024.0), 2)
    else:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        model_size_mb = round(buffer.tell() / (1024.0 * 1024.0), 2)

    return {
        "architecture": "ResNet-50",
        "pretrained_weights": getattr(model, "pretrained_weights_name", "IMAGENET1K_V2"),
        "pretrained_dataset": getattr(model, "pretrained_dataset_name", "ImageNet-1K"),
        "frozen_layers": list(getattr(model, "frozen_layers", ["conv1", "bn1", "layer1", "layer2"])),
        "trained_layers": list(getattr(model, "trained_layers", ["layer3", "layer4", "projection_head"])),
        "embedding_dim": int(getattr(model, "embedding_dim", 256)),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "model_size_mb": float(model_size_mb),
    }


def build_model_from_config(cfg: dict, pretrained: bool = True) -> ResNet50IrisEmbedder:
    """Instancia ResNet50IrisEmbedder a partir do dicionário config.yaml."""
    return ResNet50IrisEmbedder(
        embedding_dim=int(cfg.get("embedding_dim", 256)),
        freeze_up_to=str(cfg.get("freeze_up_to", "layer2")),
        pretrained=pretrained,
    )
